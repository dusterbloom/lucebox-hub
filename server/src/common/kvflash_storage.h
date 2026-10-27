// KVFlash 2.0 sessions, phase P3 (trackc/SESSIONS_DESIGN.md section 5): the storage policy over the
// session store. Policy (quota, age, action, grace, export path) persisted in policy.json; the
// tier state machine live -> disk -> compact -> exported/gone; the ask flow (a pending decision at
// quota or age, compact after the grace, delete only on an explicit yes); generation claims so a
// click, a timeout, a resume, an export and a queued writer never act on the same instance twice;
// occupancy from stat with the peak of every write reserved first; export and import.
//
// Threading: every call that reads or changes the store runs on one owner thread (the engine's,
// through the scheduler command path). Heavy copies (an export, an import's verify and copy, a log
// rewrite) run on a worker that touches only its own plan; the plan is made and the result committed
// on the owner thread, at the next pump(). The claim tables, reservations, acknowledgements and the
// finished queue are under mu_.
#pragma once

#include "kvflash_session_store.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <ctime>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

namespace luce::common {

// ---- the policy -------------------------------------------------------------------------------

struct KvFlashPolicy {
    uint64_t quota_gib = 64;              // 0: free space minus the reserve
    int64_t reserve_gib = -1;             // -1 ("auto"): min(10% of the disk, 4 GiB)
    uint64_t max_age_days = 30;           // 0: no age limit
    std::string action = "ask";           // ask | compact | export | delete-oldest
    uint64_t grace_hours = 24;            // an unanswered ask compacts after this
    std::string export_path;
    std::string after_export = "delete";  // delete | keep
    uint64_t log_dead_mib = 512;          // a disk session with this much dead log is rewritten (0: never)
    uint64_t preamble_mib = 2048;         // P1's preamble store budget, carried here so the dashboard has one object
};

inline std::string kvflash_policy_pack(const KvFlashPolicy & p) {
    std::string o = "{\"quota_gib\":" + std::to_string(p.quota_gib) + ",\"reserve_gib\":" +
                    (p.reserve_gib < 0 ? std::string("\"auto\"") : std::to_string(p.reserve_gib)) +
                    ",\"max_age_days\":" + std::to_string(p.max_age_days) + ",\"action\":";
    detail::json_str(o, p.action);
    o += ",\"grace_hours\":" + std::to_string(p.grace_hours) + ",\"export_path\":";
    detail::json_str(o, p.export_path);
    o += ",\"after_export\":";
    detail::json_str(o, p.after_export);
    o += ",\"log_dead_mib\":" + std::to_string(p.log_dead_mib) + ",\"preamble_mib\":" + std::to_string(p.preamble_mib) + "}";
    return o;
}

namespace detail {
constexpr size_t kMaxPathBytes = 4096;      // PATH_MAX
constexpr size_t kMaxIdentityBytes = 4096;  // a session id or client name from an imported meta.json

// Well-formed UTF-8: no overlong forms, no surrogates, nothing past U+10FFFF.
inline bool utf8_valid(std::string_view s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        size_t n = 0;
        uint32_t cp = 0;
        if (c < 0x80) { ++i; continue; }
        else if (c >= 0xc2 && c <= 0xdf) { n = 1; cp = c & 0x1f; }
        else if (c >= 0xe0 && c <= 0xef) { n = 2; cp = c & 0x0f; }
        else if (c >= 0xf0 && c <= 0xf4) { n = 3; cp = c & 0x07; }
        else return false;
        if (i + n >= s.size()) return false;
        for (size_t k = 1; k <= n; ++k) {
            const unsigned char d = (unsigned char)s[i + k];
            if ((d & 0xc0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3f);
        }
        if ((n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) return false;
        i += n + 1;
    }
    return true;
}
// A path an API caller names: non-empty, bounded, UTF-8, no NUL.
inline bool path_text_ok(const std::string & p) {
    return !p.empty() && p.size() <= kMaxPathBytes && p.find('\0') == std::string::npos && utf8_valid(p);
}
inline uint64_t sat_add(uint64_t a, uint64_t b) { return a > std::numeric_limits<uint64_t>::max() - b ? std::numeric_limits<uint64_t>::max() : a + b; }
// fsynced copy of one file (1 MiB at a time); errno is the failure's on false.
inline bool copy_file_synced(const std::string & from, const std::string & to, std::string & why) {
    FILE * in = std::fopen(from.c_str(), "rb");
    if (!in) { why = "cannot read " + from + ": " + std::strerror(errno); return false; }
    FILE * out = std::fopen(to.c_str(), "wb");
    if (!out) { const int e = errno; std::fclose(in); errno = e; why = "cannot write " + to + ": " + std::strerror(e); return false; }
    std::vector<char> buf(1 << 20);
    bool ok = true;
    for (size_t n; ok && (n = std::fread(buf.data(), 1, buf.size(), in)) > 0;) ok = std::fwrite(buf.data(), 1, n, out) == n;
    ok = ok && !std::ferror(in) && std::fflush(out) == 0 && ::fsync(fileno(out)) == 0;
    const int e = errno;
    std::fclose(in);
    ok = std::fclose(out) == 0 && ok;
    if (!ok) { errno = e; why = "copy of " + from + " to " + to + " failed: " + std::strerror(e); }
    return ok;
}
// An open descriptor closed with its last owner (a plan is copied between threads).
struct FdHolder {
    int fd = -1;
    explicit FdHolder(int f) : fd(f) {}
    ~FdHolder() { if (fd >= 0) ::close(fd); }
    FdHolder(const FdHolder &) = delete;
    FdHolder & operator=(const FdHolder &) = delete;
};
// The path that names an open directory for as long as it is held, whatever happens to the pathname it was opened by.
inline std::string fd_path(int fd) { return "/proc/self/fd/" + std::to_string(fd); }
// fsynced copy of `from` into a new file `name` of the directory `dir_fd` (O_EXCL, no symlink followed). `made` gets the name
// once that open has succeeded: a name that already existed is not this copy's, so it never lands there.
inline bool copy_file_at(const std::string & from, int dir_fd, const std::string & name, std::string & why, std::vector<std::string> * made = nullptr) {
    const int in = ::open(from.c_str(), O_RDONLY | O_CLOEXEC);
    if (in < 0) { why = "cannot read " + from + ": " + std::strerror(errno); return false; }
    const int out = ::openat(dir_fd, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (out < 0) { const int e = errno; ::close(in); errno = e; why = "cannot create " + name + ": " + std::strerror(e); return false; }
    if (made) made->push_back(name);
    std::vector<char> buf(1 << 20);
    bool ok = true;
    for (ssize_t n; ok && (n = ::read(in, buf.data(), buf.size())) != 0;) {
        if (n < 0) { if (errno == EINTR) continue; ok = false; break; }
        for (ssize_t at = 0; ok && at < n;) {
            const ssize_t w = ::write(out, buf.data() + at, (size_t)(n - at));
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) ok = false; else at += w;
        }
    }
    ok = ok && ::fsync(out) == 0;
    const int e = errno;
    ::close(in);
    ok = ::close(out) == 0 && ok;
    if (!ok) { errno = e; why = "copy of " + from + " to " + name + " failed: " + std::strerror(e); }
    return ok;
}
// Rename that fails if `to` exists (the kernel's RENAME_NOREPLACE where there is one).
inline int rename_noreplace_at(int dir_fd, const char * from, const char * to) {
#if defined(__linux__) && defined(SYS_renameat2)
    return (int)::syscall(SYS_renameat2, dir_fd, from, dir_fd, to, 1u /* RENAME_NOREPLACE */);
#else
    struct stat st{};
    if (::fstatat(dir_fd, to, &st, AT_SYMLINK_NOFOLLOW) == 0) { errno = EEXIST; return -1; }
    return ::renameat(dir_fd, from, dir_fd, to);
#endif
}
}  // namespace detail

// The most any policy number may be: every conversion to bytes or seconds stays under 2^63.
constexpr uint64_t kPolicyMaxGib = 1ull << 20;   // 1 PiB
constexpr uint64_t kPolicyMaxMib = 1ull << 30;   // 1 PiB
constexpr uint64_t kPolicyMaxDays = 36500;
constexpr uint64_t kPolicyMaxHours = 24 * 36500;

// A partial policy object merged over `p`; all or nothing. False with `err` naming the field. Parsed by the strict
// JSON library: every value is type-checked and bounded, escapes (surrogate pairs included) decode to UTF-8.
inline bool kvflash_policy_apply(std::string_view text, KvFlashPolicy & p, std::string & err) {
    using nlohmann::json;
    const json j = json::parse(text.begin(), text.end(), nullptr, false);
    if (j.is_discarded() || !j.is_object()) { err = "body: a JSON object is required"; return false; }
    KvFlashPolicy q = p;
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string & k = it.key();
        if (!(k == "quota_gib" || k == "reserve_gib" || k == "max_age_days" || k == "action" || k == "grace_hours" || k == "export_path" ||
              k == "after_export" || k == "log_dead_mib" || k == "preamble_mib")) {
            err = k + ": unknown field";
            return false;
        }
    }
    auto num = [&](const char * k, uint64_t & dst, uint64_t max) {
        const auto it = j.find(k);
        if (it == j.end()) return true;
        if (!it->is_number_unsigned()) { err = std::string(k) + ": a non-negative integer is required"; return false; }
        const uint64_t v = it->get<uint64_t>();
        if (v > max) { err = std::string(k) + ": at most " + std::to_string(max); return false; }
        dst = v;
        return true;
    };
    auto str = [&](const char * k, std::string & dst, std::initializer_list<const char *> allowed) {
        const auto it = j.find(k);
        if (it == j.end()) return true;
        if (!it->is_string()) { err = std::string(k) + ": a string is required"; return false; }
        const std::string v = it->get<std::string>();
        if (allowed.size() && std::none_of(allowed.begin(), allowed.end(), [&](const char * a) { return v == a; })) {
            err = std::string(k) + ": must be one of";
            for (const char * a : allowed) err += std::string(" ") + a;
            return false;
        }
        dst = v;
        return true;
    };
    if (!str("action", q.action, {"ask", "compact", "export", "delete-oldest"}) || !num("quota_gib", q.quota_gib, kPolicyMaxGib) ||
        !num("max_age_days", q.max_age_days, kPolicyMaxDays) || !num("grace_hours", q.grace_hours, kPolicyMaxHours) ||
        !str("after_export", q.after_export, {"delete", "keep"}) || !str("export_path", q.export_path, {}) ||
        !num("log_dead_mib", q.log_dead_mib, kPolicyMaxMib) || !num("preamble_mib", q.preamble_mib, kPolicyMaxMib)) return false;
    if (!q.export_path.empty() && !detail::path_text_ok(q.export_path)) { err = "export_path: not a usable path (too long, NUL or not UTF-8)"; return false; }
    if (const auto it = j.find("reserve_gib"); it != j.end()) {
        if (it->is_string() && it->get<std::string>() == "auto") q.reserve_gib = -1;
        else if (it->is_number_unsigned() && it->get<uint64_t>() <= kPolicyMaxGib) q.reserve_gib = (int64_t)it->get<uint64_t>();
        else { err = "reserve_gib: \"auto\" or a non-negative integer (at most " + std::to_string(kPolicyMaxGib) + ") is required"; return false; }
    }
    if (q.action == "export" && q.export_path.empty()) { err = "export_path: required for the export action"; return false; }
    p = std::move(q);
    return true;
}

inline bool kvflash_policy_save(const std::string & root, const KvFlashPolicy & p) {
    const std::string j = kvflash_policy_pack(p);
    return kvflash_bundle_write(root + "/policy.json", std::vector<uint8_t>(j.begin(), j.end())) && kvflash_fsync_dir(root);
}

// True when the file is absent (the defaults) or good; false when it is damaged (the defaults, kept).
inline bool kvflash_policy_load(const std::string & root, KvFlashPolicy & p) {
    std::vector<uint8_t> b;
    if (!kvflash_bundle_read(root + "/policy.json", b, 1 << 20)) return !std::filesystem::exists(root + "/policy.json");
    KvFlashPolicy q;
    std::string err;
    if (!kvflash_policy_apply(std::string_view((const char *)b.data(), b.size()), q, err)) return false;
    p = std::move(q);
    return true;
}

// ---- the tier state machine -------------------------------------------------------------------

// The state an action leads to, "" when the action is not allowed there. "parked-gpu" is any session
// in a slot (live or parked): it is never acted on from outside. A compact session comes back only
// through the client's next cold admit, never a load.
inline std::string kvflash_state_next(const std::string & state, const std::string & action) {
    if (state == "parked-gpu") return action == "evict" ? "disk" : "";
    if (state == "disk") return action == "resume" ? "parked-gpu" : action == "compact" ? "compact" : action == "export" ? "exported" : action == "delete" ? "gone" : "";
    if (state == "compact") return action == "export" ? "exported" : action == "delete" ? "gone" : "";
    if (state == "exported") return action == "import" ? "disk" : action == "delete" ? "gone" : "";
    if (state == "foreign") return action == "delete" ? "gone" : "";
    return "";
}

// ---- peaks ------------------------------------------------------------------------------------

// What a write needs on the disk at its worst: the park's records, slabs and head, plus the old log
// while a fresh log coexists with it.
inline uint64_t kvflash_park_peak(uint64_t records, uint64_t slabs, uint64_t head, uint64_t old_log_while_fresh) {
    return records + slabs + head + old_log_while_fresh;
}

// A copy-only log compaction: the live records beside the old log until the new head commits.
inline KvFlashBundleCommit kvflash_log_compaction_commit(const std::string & dir, const KvFlashSlotHead & h, size_t record_bytes, KvFlashSlotHead & nh) {
    nh = h;
    const uint32_t ng = std::max({h.gen, h.end_gen, h.ckpt_gen, h.trunk_gen, h.log_gen}) + 1;
    nh.gen = ng; nh.log_gen = ng;
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = ng; c.log_gen = ng; c.new_log = true; c.copy_from = h.log_gen; c.record_bytes = record_bytes;
    uint64_t off = kKvFlashLogHeader;
    for (size_t i = 0; i < h.map.size(); ++i)
        if (h.map[i] != 0) { c.copies.push_back(h.map[i]); nh.map[i] = off; off += record_bytes; }
    nh.log_bytes = off;
    c.head = kvflash_head_pack(nh);
    c.unlink = {kvflash_bundle_log_name(h.log_gen)};
    return c;
}

// The bytes a commit puts on the disk, exactly: the log header of a new log, every copied and every new record
// with its tag, the slab files and the new head (written beside the old one until it is renamed over it).
inline uint64_t kvflash_commit_bytes(const KvFlashBundleCommit & c) {
    uint64_t n = c.head.size();
    if (c.new_log) n += kKvFlashLogHeader + (uint64_t)c.copies.size() * c.record_bytes;
    for (const auto & r : c.records) n += r.second.size() + kKvFlashRecordTag;
    for (const auto & f : c.files) n += f.second.size();
    return n;
}

// ---- the storage ------------------------------------------------------------------------------

struct KvFlashStorageEnv {
    std::function<int64_t()> now;                          // unix seconds
    std::function<uint64_t()> free_bytes, disk_bytes;      // the filesystem under the root
    std::function<bool(const std::string &)> unlink_file;  // false: the unlink failed
    uint64_t gib = 1ull << 30, mib = 1ull << 20;           // the units of the policy (tests shrink them)
    int64_t grace_s = -1;                                  // >= 0 replaces grace_hours (LUCE_KVFLASH_GRACE_S, a test-only knob)
    int crash_at = 0;                                      // the compact transition's crash point (tests)
    bool async_export = false;                             // heavy copies run on a worker thread
    std::function<bool(const std::string &)> fsync_dir;    // directory fsync of a publication (tests inject a failure)
    std::function<void()> worker_gate;                     // runs first on every worker (tests hold the I/O here)
    std::function<void()> spawn_hook;                      // runs before a worker thread is created (tests make it throw)
    int64_t delete_hold_ms = 0;                            // a delete worker waits this long before it removes the directory (LUCE_KVFLASH_TEST_HOLD_DELETE_MS, a smoke knob)
    int64_t tick_interval_s = 30;                          // the steward's pass
    int64_t room_wait_ms = 3000;                           // how long a refused park waits for the cascade action it started (ponytail: a fixed bound, the park is refused when it passes)
    std::function<void()> commit_gap;                      // runs between the swap and the reap of commit_finished_ (tests widen the window a worker can finish in)
    std::function<int(int, const char *, const char *)> rename_noreplace;   // the no-replace rename of an export's publication (tests make it fail like drvfs); empty: the kernel's
    size_t slab_max = SIZE_MAX;                            // the largest slab file this server restores (the engine's bundle limit)
};

struct KvFlashClaim {   // what an action saw when it was offered: it proceeds only if all four still hold
    std::string key;
    uint32_t gen = 0;
    std::string state;
    int64_t last_used = 0;
};

struct KvFlashResult {
    int status = 200;   // HTTP-shaped: 200/202 ok, 400 bad request, 404 gone, 409 stale or busy, 507 no room, 500
    std::string error;
    bool ok() const { return status < 300; }
};

struct KvFlashPending {
    std::string id, key, reason, state;   // reason: quota | age | space
    uint32_t gen = 0;
    int64_t last_used = 0, created = 0, deadline = 0;
    std::vector<std::string> options;
};

// "auto" reserve: 10% of the disk, at most 4 GiB. It was max(10%, 20 GiB), 204 GiB on a 2 TB drive, which refused every
// park on a box with 28 GB free (eviction stage 0, P1).
inline uint64_t kvflash_auto_reserve_bytes(uint64_t disk_bytes, uint64_t gib) { return std::min<uint64_t>(disk_bytes / 10, 4 * gib); }

struct KvFlashOccupancy { uint64_t live = 0, dead = 0, total = 0, reserved_peaks = 0, quota = 0, free = 0; };

struct KvFlashExportPlan {
    KvFlashClaim claim;
    std::string key, src, tmp, dst, shown;   // shown: the destination as the caller named it
    std::vector<std::string> files;
    bool compact = false, delete_local = false;
    uint64_t bytes = 0, reservation = 0;
    std::shared_ptr<detail::FdHolder> dest_fd;   // the destination directory as it was checked: every later step goes through this descriptor
};

struct KvFlashImportPlan {
    std::string src, key, tmp, dst;
    std::vector<std::string> files;
    KvFlashSessionEntry entry;   // the metadata as it will be published: identity of this server, no archive links
    bool verify = false;         // a full bundle of this server: its records are read back through their crc
    uint64_t reservation = 0;
};

struct KvFlashWork { std::string kind, key, path; };   // heavy work a tick asks the owner to start: "export"

class KvFlashStorage {
public:
    KvFlashStorage(KvFlashSessionStore & store, uint64_t identity, size_t record_bytes, size_t head_max, KvFlashStorageEnv env = {})
        : store_(store), identity_(identity), rb_(record_bytes), head_max_(head_max), env_(std::move(env)) {
        const std::string root = store_.root();
        if (!env_.now) env_.now = [] { return (int64_t)std::time(nullptr); };
        if (!env_.free_bytes) env_.free_bytes = [root] { struct statvfs v{}; return ::statvfs(root.c_str(), &v) == 0 ? (uint64_t)v.f_bavail * v.f_frsize : 0; };
        if (!env_.disk_bytes) env_.disk_bytes = [root] { struct statvfs v{}; return ::statvfs(root.c_str(), &v) == 0 ? (uint64_t)v.f_blocks * v.f_frsize : 0; };
        if (!env_.unlink_file) env_.unlink_file = [](const std::string & p) { return ::unlink(p.c_str()) == 0 || errno == ENOENT; };
        if (!env_.fsync_dir) env_.fsync_dir = [](const std::string & d) { return kvflash_fsync_dir(d); };
    }
    ~KvFlashStorage() { for (auto & w : workers_) if (w.t.joinable()) w.t.join(); }
    KvFlashStorage(const KvFlashStorage &) = delete;
    KvFlashStorage & operator=(const KvFlashStorage &) = delete;

    // policy.json and pending.json; a stale <key>.tmp directory of an interrupted import goes. False
    // when a file was damaged (the defaults / no pending are used).
    bool load() {
        bool ok = kvflash_policy_load(store_.root(), policy_);
        std::vector<uint8_t> b;
        pending_.clear();
        if (kvflash_bundle_read(store_.root() + "/pending.json", b, 1 << 24)) {
            std::string text((const char *)b.data(), b.size());
            size_t at = 0;
            bool first = true;
            while (at < text.size()) {
                size_t nl = text.find('\n', at);
                if (nl == std::string::npos) nl = text.size();
                const std::string line = text.substr(at, nl - at);
                at = nl + 1;
                if (line.empty()) continue;
                std::map<std::string, std::string> m;
                if (!detail::json_flat_object(line, m)) { ok = false; pending_.clear(); break; }
                if (first) { first = false; next_id_ = std::max<uint64_t>(1, std::strtoull(m["next_id"].c_str(), nullptr, 10)); continue; }
                KvFlashPending p;
                p.id = m["id"]; p.key = m["key"]; p.reason = m["reason"]; p.state = m["state"];
                p.gen = (uint32_t)std::strtoull(m["gen"].c_str(), nullptr, 10);
                p.last_used = (int64_t)std::strtoull(m["last_used"].c_str(), nullptr, 10);
                p.created = (int64_t)std::strtoull(m["created"].c_str(), nullptr, 10);
                p.deadline = (int64_t)std::strtoull(m["deadline"].c_str(), nullptr, 10);
                for (size_t s = 0, e; s <= m["options"].size(); s = e + 1) {
                    e = m["options"].find(',', s);
                    if (e == std::string::npos) e = m["options"].size();
                    if (e > s) p.options.push_back(m["options"].substr(s, e - s));
                }
                pending_.push_back(std::move(p));
            }
        }
        std::error_code ec;
        for (const auto & de : std::filesystem::directory_iterator(store_.dir(""), ec))
            if (de.is_directory(ec) && de.path().extension() == ".tmp") std::filesystem::remove_all(de.path(), ec);
        return ok;
    }

    const KvFlashPolicy & policy() const { return policy_; }
    KvFlashSessionStore & store() { return store_; }
    bool read_head(const std::string & key, KvFlashSlotHead & h) const { return detail::read_head(dir_of_(key), identity_, head_max_, h); }
    bool set_policy(std::string_view json, std::string & err) {
        KvFlashPolicy q = policy_;
        if (!kvflash_policy_apply(json, q, err)) return false;
        if (!kvflash_policy_save(store_.root(), q)) { err = "policy.json: not written"; return false; }
        policy_ = std::move(q);
        return true;
    }

    std::vector<KvFlashPending> pending() const { return pending_; }

    // ---- claims ----

    KvFlashClaim claim_of(const std::string & key) const {
        KvFlashClaim c;
        c.key = key;
        if (const KvFlashSessionEntry * e = store_.find_key(key)) { c.gen = e->gen; c.state = e->state; c.last_used = e->last_used; }
        return c;
    }
    // Take the claim if the instance still is what the caller saw and nothing else holds it.
    KvFlashResult take(const KvFlashClaim & c) {
        const KvFlashSessionEntry * e = store_.find_key(c.key);
        if (!e) return {404, "no such session"};
        std::lock_guard<std::mutex> lk(mu_);
        if (busy_.count(c.key) || writers_[c.key] > 0) return {409, "busy: the session is being resumed, written, exported or acted on"};
        if (e->gen != c.gen || e->state != c.state || e->last_used != c.last_used) return {409, "stale: the session changed since the decision was offered"};
        busy_.insert(c.key);
        return {};
    }
    // A resume takes the claim on whatever the instance is now, and supersedes any decision about it.
    KvFlashResult claim_resume(const std::string & key) {
        KvFlashResult r = take(claim_of(key));
        if (r.ok()) drop_pending_(key);
        return r;
    }
    void release(const std::string & key) { std::lock_guard<std::mutex> lk(mu_); busy_.erase(key); }
    // A park of `key` is queued on the writer (its slot may be reused meanwhile).
    void writer_begin(const std::string & key) { std::lock_guard<std::mutex> lk(mu_); ++writers_[key]; }
    // The writer's verdict on that park (its thread): `gen` is the head generation it committed when `ok`. The key stays
    // held until the owner thread has reconciled it in apply_acks(), so no action sees a store behind the head.
    void writer_ack(const std::string & key, uint32_t gen, bool ok) {
        { std::lock_guard<std::mutex> lk(mu_); acks_.push_back({key, gen, ok}); }
        wake_();
    }
    // Owner thread: the acknowledged generation is published to the entry and its meta.json, then the key is free.
    void apply_acks() {
        std::vector<Ack> acks;
        { std::lock_guard<std::mutex> lk(mu_); acks.swap(acks_); }
        for (const Ack & a : acks) {
            for (auto & e : store_.entries())
                if (e.key == a.key && a.ok && a.gen > e.gen) {
                    e.gen = a.gen;
                    if (!store_.write_meta(a.key)) std::fprintf(stderr, "[kvflash] storage: meta.json of %s not written after park gen %u\n", a.key.c_str(), a.gen);
                }
            std::lock_guard<std::mutex> lk(mu_);
            auto it = writers_.find(a.key);
            if (it != writers_.end() && --it->second <= 0) writers_.erase(it);
        }
    }
    // The scheduler is woken (from any thread) when a worker finishes or a writer acknowledges: it must pump().
    // Replacing or clearing it waits for an invocation that is running (wake_ calls it under the lock), so once set_wake({})
    // returns nothing of the owner of the callback is reachable from a worker.
    void set_wake(std::function<void()> f) { std::lock_guard<std::mutex> lk(wake_mu_); wake_fn_ = std::move(f); }
    // Owner thread, before the store (and the directory lock beside it) is released: no work starts any more, every worker
    // is joined and its result committed while the store is alive, the acknowledgements are reconciled, the wake is disarmed.
    void shutdown() {
        { std::lock_guard<std::mutex> lk(mu_); stopping_ = true; }
        for (auto & w : workers_) if (w.t.joinable()) w.t.join();
        workers_.clear();
        commit_finished_();
        apply_acks();
        set_wake({});
    }

    // ---- compact and delete ----

    // 202 with the file work (the descriptor, the unlinks; the recursive removal) on a worker when env.async_export, committed by
    // the owner's next pump; else done. The claim is held until the commit; op_result() says how it came out.
    KvFlashResult compact(const KvFlashClaim & c) {
        KvFlashResult t = take(c);
        if (!t.ok()) return t;
        const KvFlashSessionEntry * e = store_.find_key(c.key);
        if (kvflash_state_next(e->state, "compact").empty()) { release(c.key); return {409, "a session in state " + e->state + " cannot be compacted"}; }
        const std::string key = c.key, dir = dir_of_(key);
        KvFlashSlotHead h;
        if (!detail::read_head(dir, identity_, head_max_, h)) { release(key); return {500, "head of " + key + " does not load"}; }
        note_op_(key, "compact", "running", "");
        auto done = std::make_shared<KvFlashCompactResult>();
        return run_([this, dir, h, done] { *done = kvflash_compact_dir(dir, h, env_.unlink_file, env_.crash_at); return KvFlashResult{}; },
                    [this, key, h, done](const KvFlashResult &) {
                        const KvFlashResult r = compact_commit_(key, h, *done);
                        note_op_(key, "compact", r.status >= 300 ? "failed" : "done", r.error);
                        release(key);
                        return r;
                    }, /*continuation=*/true,
                    [this, key](const KvFlashResult & r) { note_op_(key, "compact", "failed", r.error); release(key); });
    }
    KvFlashResult remove(const KvFlashClaim & c) {
        KvFlashResult t = take(c);
        if (!t.ok()) return t;
        const KvFlashSessionEntry * e = store_.find_key(c.key);
        if (kvflash_state_next(e->state, "delete").empty()) { const KvFlashResult r{409, "the session is in a slot (state " + e->state + ")"}; release(c.key); return r; }
        return delete_async_(c.key, [this, key = c.key](const KvFlashResult & r) { release(key); return r; });
    }
    // The last word on the operation a key had: op "compact" | "delete", state "running" | "done" | "failed".
    bool op_result(const std::string & key, std::string & op, std::string & state, std::string * err = nullptr) const {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = ops_.find(key);
        if (it == ops_.end()) return false;
        op = it->second.path; state = it->second.state;
        if (err) *err = it->second.error;
        return true;
    }

    // ---- export and import ----

    KvFlashResult export_begin(const KvFlashClaim & c, const std::string & path_arg, KvFlashExportPlan & plan) {
        const std::string path = path_arg.empty() ? policy_.export_path : path_arg;
        if (path.empty()) return {400, "no export path: set export_path in the policy or pass path"};
        if (!detail::path_text_ok(path)) return {400, "path: not a usable path (too long, NUL or not UTF-8)"};
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) return {400, "export path " + path + " does not exist"};
        if (!std::filesystem::is_directory(path, ec)) return {400, "export path " + path + " is not a directory"};
        if (::access(path.c_str(), W_OK | X_OK) != 0) return {400, "export path " + path + " is not writable"};
        // the destination must be independent of the store: resolved through every symlink, it may not be the store or beneath it
        const std::filesystem::path dest = std::filesystem::canonical(path, ec);
        const std::filesystem::path root = ec ? std::filesystem::path() : std::filesystem::canonical(store_.root(), ec);
        if (ec) return {400, "export path " + path + " cannot be resolved: " + ec.message()};
        if (inside_(root, dest)) return {400, "export path " + path + " is inside the managed store: the export must land outside it"};
        // hold the directory and decide on what is held: the pathname may change between here and the copy, the descriptor cannot
        auto held = std::make_shared<detail::FdHolder>(::open(dest.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        if (held->fd < 0) return {400, "export path " + path + " cannot be opened: " + std::strerror(errno)};
        const std::filesystem::path real = std::filesystem::canonical(detail::fd_path(held->fd), ec);
        if (ec || inside_(root, real)) return {400, "export path " + path + " is inside the managed store or cannot be resolved: the export must land outside it"};
        KvFlashResult t = take(c);
        if (!t.ok()) return t;
        auto fail = [&](KvFlashResult r) { release(c.key); return r; };
        const KvFlashSessionEntry * e = store_.find_key(c.key);
        if (kvflash_state_next(e->state, "export").empty()) return fail({409, "the session cannot be exported in state " + e->state});
        struct stat tmp_st{}, dst_st{};   // the staging name and the final name are both checked, on the held directory
        if (::fstatat(held->fd, (c.key + ".tmp").c_str(), &tmp_st, AT_SYMLINK_NOFOLLOW) == 0) return fail({409, "export staging " + path + "/" + c.key + ".tmp exists: remove it or choose another path"});
        if (::fstatat(held->fd, c.key.c_str(), &dst_st, AT_SYMLINK_NOFOLLOW) == 0) return fail({409, "export destination " + path + "/" + c.key + " exists"});
        KvFlashExportPlan p;
        p.claim = c; p.key = c.key; p.src = dir_of_(c.key);
        p.dst = dest.string() + "/" + c.key; p.tmp = p.dst + ".tmp"; p.shown = path + "/" + c.key; p.dest_fd = std::move(held);
        p.compact = e->state == "compact";
        p.files = {"head", "meta.json"};
        KvFlashSlotHead h;
        if (!detail::read_head(p.src, identity_, head_max_, h)) return fail({500, "head of " + c.key + " does not load"});
        if (p.compact) {
            KvFlashCompactDescriptor d;
            if (!kvflash_compact_validate(p.src, h, d)) return fail({500, "compact.json of " + c.key + " does not match its head"});
            p.files.push_back(kCompactFile);
        } else {
            p.files.push_back(kvflash_bundle_log_name(h.log_gen));
            for (uint32_t id : {h.end_gen, h.ckpt_gen, h.trunk_gen}) if (id) p.files.push_back(kvflash_bundle_slab_name(id));
        }
        for (const auto & f : p.files) p.bytes += detail::file_size_or0(p.src + "/" + f);
        p.delete_local = policy_.after_export == "delete";
        struct stat a{}, b{};
        if (::stat(p.src.c_str(), &a) == 0 && ::stat(dest.c_str(), &b) == 0 && a.st_dev == b.st_dev) {   // same filesystem: the copy needs the session's bytes again
            std::string why;
            if (!reserve(p.bytes, p.reservation, why, /*counts_for_quota=*/false)) return fail({507, why});
        }
        plan = std::move(p);
        return {};
    }
    // The copy: no store access, safe on a worker. <key>.tmp/ is created inside the destination directory the plan holds (it
    // must not exist: a directory the export did not make is never touched), written, every record's crc is verified on the
    // copy, files and directory are fsynced, then renamed to <key>/ (never over anything) and the destination fsynced:
    // success means the export is durable. Every step goes through the held descriptors, so no pathname can redirect it.
    KvFlashResult export_copy(const KvFlashExportPlan & p) const {
        const int dfd = p.dest_fd ? p.dest_fd->fd : -1;
        const std::string name = p.key + ".tmp";
        if (dfd < 0) return {500, "export has no destination"};
        if (::mkdirat(dfd, name.c_str(), 0755) != 0) return {errno == ENOSPC ? 507 : errno == EEXIST ? 409 : 500, "cannot create " + p.tmp + ": " + std::strerror(errno)};
        const detail::FdHolder stage(::openat(dfd, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
        std::vector<std::string> made;   // what this export created in the directory: all it ever removes
        bool published = false;
        auto withdraw = [&] {
            if (stage.fd >= 0) for (const auto & f : made) ::unlinkat(stage.fd, f.c_str(), 0);
            ::unlinkat(dfd, published ? p.key.c_str() : name.c_str(), AT_REMOVEDIR);
        };
        auto fail = [&](KvFlashResult r) { withdraw(); return r; };
        if (stage.fd < 0) return fail({500, "cannot open " + p.tmp});
        for (const auto & f : p.files) {
            std::string why;
            if (!detail::copy_file_at(p.src + "/" + f, stage.fd, f, why, &made)) return fail({errno == ENOSPC ? 507 : 500, why});
        }
        const std::string staged = detail::fd_path(stage.fd);
        if (!p.compact) {
            std::string why;
            if (!verify_bundle_dir_(staged, why)) return fail({500, "the copy does not verify: " + why});
        }
        if (!env_.fsync_dir(staged)) return fail({500, "fsync of " + p.tmp + " failed"});
        if ((env_.rename_noreplace ? env_.rename_noreplace(dfd, name.c_str(), p.key.c_str()) : detail::rename_noreplace_at(dfd, name.c_str(), p.key.c_str())) != 0) {
            const int err = errno;
            if (err != EINVAL && err != ENOSYS && err != EOPNOTSUPP) return fail({500, "rename to " + p.dst + ": " + std::strerror(err)});
            withdraw();   // no RENAME_NOREPLACE here (drvfs on WSL): the staged copy goes, the export is made in place
            return export_copy_in_place_(p, dfd);
        }
        published = true;
        // the entry now named <key> must be the directory that was verified: a staging name swapped for another directory before
        // the rename publishes something else, which this export neither deletes the source for nor removes
        struct stat want{}, got{};
        if (::fstat(stage.fd, &want) != 0 || ::fstatat(dfd, p.key.c_str(), &got, AT_SYMLINK_NOFOLLOW) != 0 || want.st_dev != got.st_dev || want.st_ino != got.st_ino) {
            for (const auto & f : made) ::unlinkat(stage.fd, f.c_str(), 0);   // only the directory this export verified, by descriptor
            return {500, "the published " + p.dst + " is not the directory that was verified: the export failed, the source is kept"};
        }
        if (!env_.fsync_dir(detail::fd_path(dfd))) {   // the entry may not survive a power loss: the source stays, the copy is withdrawn
            withdraw();
            return {500, "fsync of " + std::filesystem::path(p.dst).parent_path().string() + " failed after the rename: the export is not durable, the source is kept"};
        }
        return {};
    }
    // The export on a filesystem that cannot rename without replacing: the final directory is created exclusively (mkdirat), the files
    // are copied through the held descriptors (O_EXCL, no symlink) with `head` last, the copy is verified, and the files, the directory
    // and its parent are fsynced before success, which is what lets the caller delete the source. Any failure removes what this copy
    // made and keeps the source. Interrupted at any point it is a directory with no head, which import does not take for an export.
    // ponytail: the copy is made a second time after the staged one failed to publish; a per-destination memory of "no rename" would skip it.
    KvFlashResult export_copy_in_place_(const KvFlashExportPlan & p, int dfd) const {
        if (::mkdirat(dfd, p.key.c_str(), 0755) != 0) return {errno == ENOSPC ? 507 : errno == EEXIST ? 409 : 500, "cannot create " + p.dst + ": " + std::strerror(errno)};
        const detail::FdHolder dir(::openat(dfd, p.key.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
        std::vector<std::string> made;
        auto fail = [&](KvFlashResult r) {
            if (dir.fd >= 0) for (const auto & f : made) ::unlinkat(dir.fd, f.c_str(), 0);
            ::unlinkat(dfd, p.key.c_str(), AT_REMOVEDIR);
            return r;
        };
        if (dir.fd < 0) return fail({500, "cannot open " + p.dst});
        std::vector<std::string> order;
        for (const auto & f : p.files) if (f != "head") order.push_back(f);
        order.push_back("head");
        for (const auto & f : order) {
            std::string why;
            if (!detail::copy_file_at(p.src + "/" + f, dir.fd, f, why, &made)) return fail({errno == ENOSPC ? 507 : 500, why});
        }
        const std::string made_path = detail::fd_path(dir.fd);
        std::string why;
        if (!p.compact && !verify_bundle_dir_(made_path, why)) return fail({500, "the copy does not verify: " + why});
        if (!env_.fsync_dir(made_path)) return fail({500, "fsync of " + p.dst + " failed"});
        if (!env_.fsync_dir(detail::fd_path(dfd))) return fail({500, "fsync of " + std::filesystem::path(p.dst).parent_path().string() + " failed: the export is not durable, the source is kept"});
        return {};
    }
    // The copy's verdict is recorded; the local session is removed only after a verified, durable copy, and that removal runs on a
    // worker too (the key stays claimed until it is done).
    KvFlashResult export_commit(const KvFlashExportPlan & p, const KvFlashResult & copied) {
        if (p.reservation) release_reservation(p.reservation);
        auto record = [this, p, copied](const KvFlashResult * removed) {
            ExportRecord rec{p.shown, copied.ok() ? (p.delete_local ? "exported" : "copied") : "failed", copied.error};
            KvFlashResult r = copied;
            if (removed && !removed->ok()) { rec.state = "copied"; rec.error = "the copy is durable but the local session was not removed: " + removed->error; r = *removed; }
            { std::lock_guard<std::mutex> lk(mu_); exports_[p.key] = rec; }
            release(p.key);
            return r;
        };
        if (!copied.ok() || !p.delete_local) return record(nullptr);
        return delete_async_(p.key, [record](const KvFlashResult & d) { return record(&d); });
    }
    KvFlashResult export_session(const KvFlashClaim & c, const std::string & path) {
        KvFlashExportPlan plan;
        KvFlashResult r = export_begin(c, path, plan);
        if (!r.ok()) return r;
        return export_commit(plan, export_copy(plan));
    }
    // 202 with the copy on a worker (committed by the owner's next pump) when env.async_export, else done.
    KvFlashResult export_async(const KvFlashClaim & c, const std::string & path) {
        KvFlashExportPlan plan;
        KvFlashResult r = export_begin(c, path, plan);
        if (!r.ok()) return r;
        return run_([this, plan] { return export_copy(plan); }, [this, plan](const KvFlashResult & copied) { return export_commit(plan, copied); }, false,
                    [this, plan](const KvFlashResult & refused) { export_commit(plan, refused); });   // recorded failed, reservation and claim given back
    }
    bool export_result(const std::string & key, std::string & path, std::string & state, std::string * err = nullptr) const {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = exports_.find(key);
        if (it == exports_.end()) return false;
        path = it->second.path; state = it->second.state;
        if (err) *err = it->second.error;
        return true;
    }

    // "importing" while the worker copies, then "imported" or "failed".
    bool import_result(const std::string & key, std::string & state, std::string * err = nullptr) const {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = imports_.find(key);
        if (it == imports_.end()) return false;
        state = it->second.state;
        if (err) *err = it->second.error;
        return true;
    }

    // `path` is an exported session directory. Never overwrites: it lands under a key of its own. The plan is made
    // here (identity, metadata, key, reservation); verification and copies run on a worker; the entry is published
    // on the owner thread.
    KvFlashResult import_begin(const std::string & path, bool as_foreign, KvFlashImportPlan & plan) {
        if (!detail::path_text_ok(path)) return {400, "path: not a usable path (empty, too long, NUL or not UTF-8)"};
        std::error_code ec;
        if (!std::filesystem::is_directory(path, ec)) return {400, path + " is not a directory"};
        if (!std::filesystem::exists(path + "/head", ec)) return {400, path + " holds no session (no head)"};
        std::vector<uint8_t> mb;
        KvFlashSessionEntry e;
        if (!kvflash_bundle_read(path + "/meta.json", mb, 1 << 20) || !kvflash_meta_unpack(std::string_view((const char *)mb.data(), mb.size()), e))
            return {400, path + " has no readable meta.json"};
        if (e.sid.size() > detail::kMaxIdentityBytes || e.client.size() > detail::kMaxIdentityBytes || !detail::utf8_valid(e.sid) || !detail::utf8_valid(e.client))
            return {400, path + " names a session identity that is too long or not UTF-8"};
        KvFlashSlotHead h;
        bool ours = detail::read_head(path, identity_, head_max_, h);
        const bool marker = std::filesystem::symlink_status(path + "/" + kCompactFile, ec).type() != std::filesystem::file_type::not_found;
        bool compact = false;
        if (ours && marker) {
            KvFlashCompactDescriptor d;
            compact = kvflash_compact_validate(path, h, d);
            if (!compact) {
                if (!as_foreign) return {400, path + ": compact.json does not match the head"};
                ours = false;   // an operator keeps it as a foreign record, every file intact
            }
        }
        if (!ours && !as_foreign) {
            char b[96];
            std::snprintf(b, sizeof b, "identity mismatch: the head is of identity %016llx, this server is %016llx",
                          (unsigned long long)detail::peek_head_identity(path), (unsigned long long)identity_);
            return {409, b};
        }
        std::vector<std::string> files;
        if (!ours) { for (const auto & de : std::filesystem::directory_iterator(path, ec)) if (de.is_regular_file(ec) && kvflash_safe_file_name(de.path().filename().string())) files.push_back(de.path().filename().string()); }
        else {
            files = {"head", "meta.json"};
            if (compact) files.push_back(kCompactFile);
            else {
                files.push_back(kvflash_bundle_log_name(h.log_gen));
                for (uint32_t id : {h.end_gen, h.ckpt_gen, h.trunk_gen}) if (id) files.push_back(kvflash_bundle_slab_name(id));
            }
        }
        uint64_t bytes = 0;
        for (const auto & f : files) bytes += detail::file_size_or0(path + "/" + f);
        KvFlashImportPlan p;
        std::string why;
        if (!reserve(detail::sat_add(bytes, kvflash_meta_pack(e).size()), p.reservation, why)) return {507, why};
        // the key is free in the store, among the keys held for imports in flight and on the disk; held until the import completes
        std::set<std::string> on_disk;
        for (const auto & de : std::filesystem::directory_iterator(store_.dir(""), ec)) on_disk.insert(de.path().filename().string());
        p.key = store_.reserve_key(e.client, e.sid, std::move(on_disk));
        p.src = path; p.files = std::move(files);
        p.dst = store_.dir(p.key); p.tmp = p.dst + ".tmp";
        p.verify = ours && !compact;
        e.key = p.key; e.state = !ours ? "foreign" : compact ? "compact" : "disk"; e.identity = ours ? identity_ : detail::peek_head_identity(path);
        e.archive_of.clear(); e.archive.clear(); e.bytes_dead = 0;
        if (ours) { e.tokens = h.history.size(); e.gen = h.gen; if (!compact) e.history = std::move(h.history); }
        p.entry = std::move(e);
        { std::lock_guard<std::mutex> lk(mu_); imports_[p.key] = {"", "importing", ""}; }
        plan = std::move(p);
        return {};
    }
    // The worker's part: copy into <key>.tmp/, verify the copy, write the normalized meta.json, fsync, and only then
    // publish the directory (rename + parent fsync). Touches no store state.
    KvFlashResult import_copy(const KvFlashImportPlan & p) const {
        std::error_code ec;
        std::filesystem::remove_all(p.tmp, ec);
        if (::mkdir(p.tmp.c_str(), 0755) != 0) return {500, "cannot create " + p.tmp + ": " + std::strerror(errno)};
        auto fail = [&](KvFlashResult r) { std::filesystem::remove_all(p.tmp, ec); return r; };
        std::string why;
        for (const auto & f : p.files) {
            if (f == "meta.json") continue;   // replaced below by the normalized one
            if (!detail::copy_file_synced(p.src + "/" + f, p.tmp + "/" + f, why)) return fail({errno == ENOSPC ? 507 : 500, why});
        }
        if (p.verify && !verify_bundle_dir_(p.tmp, why)) return fail({400, "the bundle does not verify: " + why});
        KvFlashSessionEntry e = p.entry;
        e.bytes_live = kvflash_tree_bytes(p.tmp);
        const std::string meta = kvflash_meta_pack(e);
        if (!kvflash_bundle_write(p.tmp + "/meta.json", std::vector<uint8_t>(meta.begin(), meta.end()))) return fail({500, "meta.json of the import not written"});
        if (!env_.fsync_dir(p.tmp)) return fail({500, "fsync of " + p.tmp + " failed"});
        if (std::rename(p.tmp.c_str(), p.dst.c_str()) != 0) return fail({500, "cannot publish " + p.dst + ": " + std::strerror(errno)});
        const std::string parent = std::filesystem::path(p.dst).parent_path().string();
        if (!env_.fsync_dir(parent)) {   // not durable: withdraw it rather than list a session that may vanish
            std::filesystem::remove_all(p.dst, ec);
            return {500, "fsync of " + parent + " failed after publishing the import"};
        }
        return {};
    }
    KvFlashResult import_commit(const KvFlashImportPlan & p, const KvFlashResult & copied) {
        release_reservation(p.reservation);
        store_.release_key(p.key);
        if (!copied.ok()) {
            std::error_code ec;
            std::filesystem::remove_all(p.tmp, ec);
            std::lock_guard<std::mutex> lk(mu_);
            imports_[p.key] = {"", "failed", copied.error};
            return copied;
        }
        KvFlashSessionEntry e = p.entry;
        e.bytes_live = kvflash_tree_bytes(p.dst);
        if (!store_.insert_new(std::move(e))) {   // never over a live entry: the directory is left as it is
            const KvFlashResult r{409, "key " + p.key + " is taken by a live session: the import is not listed"};
            std::lock_guard<std::mutex> lk(mu_);
            imports_[p.key] = {"", "failed", r.error};
            return r;
        }
        { std::lock_guard<std::mutex> lk(mu_); imports_[p.key] = {"", "imported", ""}; }
        KvFlashResult r;
        if (!store_.write_index()) r.error = "index.json not written (the scan rebuilds it)";
        return r;
    }
    // `reserved` hears the key the moment it is held, before any copy: a caller that stops waiting can still name the import.
    KvFlashResult import_dir(const std::string & path, bool as_foreign, std::string * key_out, const std::function<void(const std::string &)> & reserved = {}) {
        KvFlashImportPlan plan;
        KvFlashResult r = import_begin(path, as_foreign, plan);
        if (!r.ok()) return r;
        if (key_out) *key_out = plan.key;
        if (reserved) reserved(plan.key);
        return run_([this, plan] { return import_copy(plan); }, [this, plan](const KvFlashResult & copied) { return import_commit(plan, copied); }, false,
                    [this, plan](const KvFlashResult &) {   // refused: the key and the bytes held for it go back, nothing was started
                        release_reservation(plan.reservation);
                        store_.release_key(plan.key);
                        std::lock_guard<std::mutex> lk(mu_);
                        imports_.erase(plan.key);
                    });
    }

    // ---- pending decisions ----

    KvFlashResult answer(const std::string & id, const std::string & action, const std::string & path) {
        const KvFlashPending * found = nullptr;
        for (const auto & p : pending_) if (p.id == id) found = &p;
        if (!found) {   // an id that was issued and is gone (answered, superseded by a resume) is stale, one never issued is unknown
            const uint64_t n = id.size() > 1 && id[0] == 'p' ? std::strtoull(id.c_str() + 1, nullptr, 10) : 0;
            return n >= 1 && n < next_id_ ? KvFlashResult{409, "stale: the decision was answered or superseded"} : KvFlashResult{404, "no such pending decision"};
        }
        const KvFlashPending p = *found;
        if (action != "compact" && action != "export" && action != "delete") return {400, "action: compact, export or delete"};
        if (std::find(p.options.begin(), p.options.end(), action) == p.options.end()) return {400, "action " + action + " is not on offer for this decision"};
        if (action == "export" && path.empty() && policy_.export_path.empty()) return {400, "no export path: set export_path in the policy or pass path"};
        const KvFlashClaim claim{p.key, p.gen, p.state, p.last_used};
        KvFlashResult r = action == "compact" ? compact(claim) : action == "delete" ? remove(claim) : export_async(claim, path);
        if (r.status == 404 || (r.status == 409 && r.error.rfind("stale", 0) == 0)) { drop_pending_(p.key); r.status = 409; }
        else if (r.ok()) drop_pending_(p.key);
        return r;
    }

    // ---- occupancy and reservation ----

    uint64_t reserve_bytes() const {
        return policy_.reserve_gib >= 0 ? (uint64_t)policy_.reserve_gib * env_.gib : kvflash_auto_reserve_bytes(env_.disk_bytes(), env_.gib);
    }
    uint64_t quota_bytes(uint64_t total, uint64_t free) const {
        if (policy_.quota_gib > 0) return policy_.quota_gib * env_.gib;
        const uint64_t room = detail::sat_add(total, free), reserve = reserve_bytes();
        return room > reserve ? room - reserve : 0;
    }
    // ponytail: occupancy re-walks the whole root on every call (stat only, a few thousand files); a per-key cache refreshed after each ack comes if the walk shows up in a profile
    KvFlashOccupancy occupancy() const {
        KvFlashOccupancy o;
        for (const auto & e : store_.entries()) { o.live += e.bytes_live; o.dead += e.bytes_dead; }
        o.total = kvflash_tree_bytes(store_.root());
        o.free = env_.free_bytes();
        o.quota = quota_bytes(o.total, o.free);
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto & r : reserved_) o.reserved_peaks = detail::sat_add(o.reserved_peaks, r.second);
        return o;
    }
    // Reserve the peak of a write before it is scheduled: free - reserve - reserved peaks >= peak, and, for a write that
    // grows the store, the store plus every reserved write plus this one stays under 1.1 x quota. A refusal means the
    // write is not scheduled (a park is skipped, the session stays in its slot).
    bool reserve(uint64_t peak, uint64_t & id, std::string & why, bool counts_for_quota = true) {
        const uint64_t free = env_.free_bytes(), floor = reserve_bytes();
        const bool grows = policy_.quota_gib > 0 && counts_for_quota && peak > 0;
        const uint64_t total = grows ? kvflash_tree_bytes(store_.root()) : 0;
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t held = 0;
        for (const auto & r : reserved_) held = detail::sat_add(held, r.second);
        if (grows) {
            const uint64_t limit = policy_.quota_gib * env_.gib / 10 * 11;
            // the first write is held to the limit like every later one: the store, the queued writes and this one stay under it
            if (total > limit || detail::sat_add(detail::sat_add(total, held), peak) > limit) {
                why = "the store (" + std::to_string(total) + " bytes) with the queued writes (" + std::to_string(held) + ") and this one (" + std::to_string(peak) +
                      ") would pass 1.1 x quota (" + std::to_string(limit) + "): new durable growth is refused";
                return false;
            }
        }
        if (free < detail::sat_add(detail::sat_add(floor, held), peak)) {
            why = "free " + std::to_string(free) + " is below the reserve " + std::to_string(floor) + " + reserved peaks " + std::to_string(held) + " + this write's peak " + std::to_string(peak);
            return false;
        }
        id = ++next_res_;
        reserved_[id] = peak;
        return true;
    }
    // The peak of one park, from the commit the engine built: its exact bytes, plus the replacement meta.json and index.json
    // that are written beside the old ones.
    bool reserve_commit(const KvFlashBundleCommit & c, const KvFlashSessionEntry & e, uint64_t & id, std::string & why) {
        const uint64_t meta = kvflash_meta_pack(e).size();
        return reserve(detail::sat_add(kvflash_commit_bytes(c), detail::sat_add(meta, detail::sat_add(store_.index_json().size(), meta))), id, why);
    }
    void release_reservation(uint64_t id) { std::lock_guard<std::mutex> lk(mu_); reserved_.erase(id); }
    // A park refused for disk room asks the steward for room, by the policy's action on the oldest disk sessions: delete-oldest
    // removes, compact compacts, export exports; the park waits only for the claim of the action it started (never for another
    // worker), at most room_wait_ms, then the reservation is retried. ask queues one "space" question and the park stays refused.
    // A session that is parked, held, busy or the parker itself is never a candidate (the tick's own filter). A refused
    // action is logged and the last refusal ends up in `why`; an action still running after the bound refuses this park and
    // finishes in the background, so the next park finds the room. ponytail: the wait is a fixed bound on the scheduler thread; a
    // deferred park retried by the pump would remove it.
    bool reserve_commit_with_room(const KvFlashBundleCommit & c, const KvFlashSessionEntry & e, uint64_t & id, std::string & why) {
        if (reserve_commit(c, e, id, why)) return true;
        std::vector<KvFlashSessionEntry> cand;
        for (const auto & s : store_.entries())
            if ((s.state == "disk" || s.state == "compact") && s.key != e.key && !is_held_(s.key) && !exempt_(s)) cand.push_back(s);
        std::stable_sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.last_used < b.last_used; });
        const std::string & act = policy_.action;
        if (act == "ask") {
            const std::string q = ask_space_(cand.empty() ? nullptr : &cand.front(), env_.now());
            if (!q.empty()) why += "; " + q;
            return false;
        }
        std::string last;
        for (const auto & s : cand) {
            const KvFlashClaim claim = claim_of(s.key);
            if (act == "compact" && s.state != "disk") continue;
            const KvFlashResult r = act == "delete-oldest" ? remove(claim) : act == "compact" ? compact(claim) : export_async(claim, policy_.export_path);
            if (!r.ok()) {
                last = act + " " + s.key + ": " + std::to_string(r.status) + " " + r.error;
                std::fprintf(stderr, "[kvflash] storage: %s of %s for a park refused: %d %s\n", act.c_str(), s.key.c_str(), r.status, r.error.c_str());
                continue;
            }
            std::fprintf(stderr, "[kvflash] storage: %s %s to make room for a park\n", act.c_str(), s.key.c_str());
            if (!wait_released_(s.key)) {
                why += "; " + act + " of " + s.key + " is still running after " + std::to_string(env_.room_wait_ms) + " ms: this park is refused, the action finishes in the background";
                std::fprintf(stderr, "[kvflash] storage: %s of %s still running after %lld ms, park refused\n", act.c_str(), s.key.c_str(), (long long)env_.room_wait_ms);
                return false;
            }
            if (reserve_commit(c, e, id, why)) return true;
        }
        if (!last.empty()) why += "; last cascade refusal: " + last;
        return false;
    }

    // ---- the steward ----

    // The scheduler's step, every iteration: acknowledgements and finished workers are committed; the steward runs
    // when its interval is up or a decision's deadline has passed. Exports it wants start on their worker.
    void pump() {
        apply_acks();
        commit_finished_();
        const int64_t now = env_.now();
        if (now - tick_at_ < env_.tick_interval_s && !deadline_passed_(now)) return;
        tick_at_ = now;
        for (const KvFlashWork & w : tick()) {
            const KvFlashResult r = export_async(claim_of(w.key), w.path);
            if (!r.ok() && r.status != 202) std::fprintf(stderr, "[kvflash] storage: export of %s not started: %s\n", w.key.c_str(), r.error.c_str());
        }
    }
    // Milliseconds until pump() has something to do: 0 for work waiting now, else the sooner of the interval and the
    // next decision's deadline.
    int64_t wake_ms() const {
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!finished_.empty() || !acks_.empty()) return 0;
        }
        const int64_t now = env_.now();
        int64_t due = tick_at_ + env_.tick_interval_s;
        for (const KvFlashPending & p : pending_)
            if (p.deadline > tick_at_ && !is_held_(p.key)) due = std::min(due, p.deadline);
        return due <= now ? 0 : (due - now) * 1000;
    }

    // One pass: finish exports, drop stale decisions, compact what passed its grace, rewrite dead logs, act on quota and age by
    // the policy, then retry the unlinks the descriptors list. Returns the exports it wants started.
    std::vector<KvFlashWork> tick() {
        std::vector<KvFlashWork> work;
        apply_acks();
        commit_finished_();
        const int64_t now = env_.now();
        // decisions whose instance moved on are void; an expired one compacts, never deletes
        for (const KvFlashPending & p : pending()) {
            const KvFlashSessionEntry * e = store_.find_key(p.key);
            if (!e || exempt_(*e)) { drop_pending_(p.key); continue; }   // gone, or an archive since the decision was made
            const bool held = is_held_(p.key);
            if (!held && (e->gen != p.gen || e->state != p.state || e->last_used != p.last_used)) { drop_pending_(p.key); continue; }
            if (held || p.deadline > now) continue;
            if (has_option_(p, "compact") && compact(KvFlashClaim{p.key, p.gen, p.state, p.last_used}).ok()) drop_pending_(p.key);
        }
        reclaim_logs_(now);
        const uint64_t total = kvflash_tree_bytes(store_.root()), free = env_.free_bytes(), quota = quota_bytes(total, free);
        const bool over = total > quota, floor_hit = (policy_.quota_gib > 0 && total > quota / 10 * 11) || free < reserve_bytes();
        std::vector<KvFlashSessionEntry> cand;   // oldest first
        for (const auto & e : store_.entries())
            if ((e.state == "disk" || e.state == "compact") && !is_held_(e.key) && !exempt_(e)) cand.push_back(e);
        std::stable_sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.last_used < b.last_used; });
        const std::string & act = policy_.action;
        // compaction before anything destructive: a compact session is reached only once every disk one is spoken for,
        // unless the policy deletes the oldest or exports, which do not care about the tier
        if (act != "delete-oldest" && act != "export")
            std::stable_partition(cand.begin(), cand.end(), [](const auto & e) { return e.state == "disk"; });
        const auto is_aged = [&](const KvFlashSessionEntry & e) { return policy_.max_age_days > 0 && now - e.last_used > (int64_t)policy_.max_age_days * 86400; };
        int64_t need = over ? (int64_t)(total - quota) : 0;
        for (const auto & e : cand) {
            const bool by_age = is_aged(e), by_quota = need > 0;
            if (!by_age && !by_quota) continue;
            const std::string reason = by_quota ? "quota" : "age";
            const int64_t bytes = (int64_t)kvflash_tree_bytes(dir_of_(e.key));
            const KvFlashClaim claim = claim_of(e.key);
            // ponytail: a compaction frees the listed files and adds a descriptor of about 256 bytes; ask counts that, then the answer's real effect shows at the next pass
            int64_t compact_saves = 0;
            if (e.state == "disk") for (const auto & f : kvflash_compact_leftovers(dir_of_(e.key))) compact_saves += (int64_t)detail::file_size_or0(dir_of_(e.key) + "/" + f);
            compact_saves = std::max<int64_t>(0, compact_saves - 256);
            if (act == "delete-oldest") {
                if (remove(claim).ok()) { std::fprintf(stderr, "[kvflash] storage: deleted %s (%s, policy delete-oldest)\n", e.key.c_str(), reason.c_str()); need -= bytes; }
            } else if (act == "compact" && e.state == "disk") {
                const KvFlashResult r = compact(claim);   // 202: the unlinks are on a worker, so what it frees is the estimate
                if (r.ok()) need -= r.status == 202 ? compact_saves : std::max<int64_t>(0, bytes - (int64_t)kvflash_tree_bytes(dir_of_(e.key)));
            } else if (act == "export") {
                work.push_back({"export", e.key, policy_.export_path});
                need -= bytes;
            } else {   // ask, or compact with nothing left to compact: a decision
                if (!has_pending_(e.key)) add_pending_(e, reason, now);
                need -= e.state == "disk" ? compact_saves : bytes;
            }
        }
        // past the hard floor the oldest decision is compacted at once
        if (floor_hit)
            for (const KvFlashPending & p : pending())
                if (has_option_(p, "compact") && !is_held_(p.key)) {
                    std::fprintf(stderr, "[kvflash] storage: past the hard floor, compacting %s now\n", p.key.c_str());
                    if (compact(KvFlashClaim{p.key, p.gen, p.state, p.last_used}).ok()) { drop_pending_(p.key); break; }
                }
        retry_unlinks_(work);   // after the policy: a retry never takes the claim of a session the policy picked, however long an unlink keeps failing
        return work;
    }

private:
    struct ExportRecord { std::string path, state, error; };
    struct Ack { std::string key; uint32_t gen; bool ok; };
    struct Worker { std::thread t; std::shared_ptr<std::atomic<bool>> done; };

    void wake_() {
        std::lock_guard<std::mutex> lk(wake_mu_);   // held across the call: set_wake({}) cannot return while it runs
        if (wake_fn_) wake_fn_();
    }
    // Run `work` (no store access) on a worker and hand its result to `commit` on the owner thread at the next pump:
    // 202 now. Without a worker (env.async_export false) both run here and the commit's result is the answer.
    // `continuation`: the second stage of an operation already under way (a removal after an export's copy): after shutdown it
    // runs here, on the owner thread, instead of being refused. A refusal (shutting down, no thread to be had) calls `refused`
    // with the verdict, so the caller gives back what it took (its claim, its reservation), and is returned as the answer.
    template <class Work, class Commit, class Refused>
    KvFlashResult run_(Work work, Commit commit, bool continuation, Refused refused) {
        bool stopping;
        { std::lock_guard<std::mutex> lk(mu_); stopping = stopping_; }
        if (stopping && !continuation) { const KvFlashResult r{503, "storage is shutting down"}; refused(r); return r; }
        if (!env_.async_export || stopping) return commit(work());
        reap_workers_();
        auto done = std::make_shared<std::atomic<bool>>(false);
        try {
            if (env_.spawn_hook) env_.spawn_hook();
            std::thread t([this, work, commit, done] {
                if (env_.worker_gate) env_.worker_gate();
                const KvFlashResult r = work();
                { std::lock_guard<std::mutex> lk(mu_); finished_.push_back([commit, r]() mutable { commit(r); }); }
                done->store(true);
                wake_();
            });
            workers_.push_back({std::move(t), done});
        } catch (const std::exception & ex) {
            const KvFlashResult r{503, std::string("cannot start a storage worker: ") + ex.what()};
            refused(r);
            return r;
        }
        return {202, ""};
    }
    template <class Work, class Commit>
    KvFlashResult run_(Work work, Commit commit, bool continuation = false) { return run_(std::move(work), std::move(commit), continuation, [](const KvFlashResult &) {}); }
    void reap_workers_() {
        for (auto it = workers_.begin(); it != workers_.end();) {
            if (it->done->load()) { it->t.join(); it = workers_.erase(it); }
            else ++it;
        }
    }
    // The claim on `key` is given back (the action's last commit ran), the owner thread committing finished workers meanwhile; false
    // when room_wait_ms pass first. Only this key is waited for: an unrelated import, export or rewrite is committed when it finishes.
    bool wait_released_(const std::string & key) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(env_.room_wait_ms);
        for (;;) {
            commit_finished_();
            if (!is_held_(key)) return true;
            if (std::chrono::steady_clock::now() >= end) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    // At most one "space" question is open, ever (the cascade and the dead-log pass share it); a session that already carries a
    // question of any reason is not asked about twice. The text names the question for the log line.
    std::string ask_space_(const KvFlashSessionEntry * e, int64_t now) {
        for (const auto & p : pending_) if (p.reason == "space") return "question " + p.id + " (space) is waiting for an answer";
        if (!e) return "";
        for (const auto & p : pending_) if (p.key == e->key) return "question " + p.id + " (" + p.reason + ") about " + e->key + " is waiting for an answer";
        add_pending_(*e, "space", now);
        return "asked " + pending_.back().id + " (space) about " + e->key;
    }
    void commit_finished_() {
        std::vector<std::function<void()>> done;
        { std::lock_guard<std::mutex> lk(mu_); done.swap(finished_); }
        if (env_.commit_gap) env_.commit_gap();
        for (auto & f : done) f();
        reap_workers_();
    }
    bool deadline_passed_(int64_t now) const {
        for (const KvFlashPending & p : pending_)
            if (p.deadline > tick_at_ && p.deadline <= now && !is_held_(p.key)) return true;
        return false;
    }
    // `p` is `root` or beneath it (both canonical)
    static bool inside_(const std::filesystem::path & root, const std::filesystem::path & p) {
        const std::string r = root.string(), q = p.string();
        return q == r || (q.size() > r.size() && q.compare(0, r.size(), r) == 0 && (r.back() == '/' || q[r.size()] == '/'));
    }

    bool is_held_(const std::string & key) const { std::lock_guard<std::mutex> lk(mu_); auto w = writers_.find(key); return busy_.count(key) || (w != writers_.end() && w->second > 0); }
    bool has_pending_(const std::string & key) const { for (const auto & p : pending_) if (p.key == key) return true; return false; }
    static bool has_option_(const KvFlashPending & p, const char * o) { return std::find(p.options.begin(), p.options.end(), o) != p.options.end(); }
    // an old instance whose successor is live is the live session's archive: not aged out from under it. From the admission
    // that evicted it the successor names it (archive_of) while its own first park, which publishes `archive`, is still to come.
    // Contract (coordinator ruling, stage 3 review P1): only the live session's immediate archive is exempt, not the chain behind it: archives are
    // independent directories, and a chain pinned through many harness compactions would let one session defeat the quota.
    bool exempt_(const KvFlashSessionEntry & e) const {
        if (!e.archive.empty()) {
            const KvFlashSessionEntry * s = store_.find_key(e.archive);
            return s && s->state == "parked-gpu";
        }
        for (const KvFlashSessionEntry & s : store_.entries()) if (s.archive_of == e.key && s.state == "parked-gpu") return true;
        return false;
    }
    std::string dir_of_(const std::string & key) const {   // a foreign slot<i>/ the migration left in place lives at the root
        return detail::is_slot_dir_name(key) ? store_.root() + "/" + key : store_.dir(key);
    }

    // The directory's removal (the recursive part) is the worker's; `then` gets the verdict on the owner thread once the entry,
    // the decision and the unlink debt are gone with it, so nothing owed to this incarnation can reach a later one under the
    // same key. The caller holds the claim and `then` releases it.
    template <class Then>
    KvFlashResult delete_async_(const std::string & key, Then then) {
        const std::string dir = dir_of_(key);
        note_op_(key, "delete", "running", "");
        auto gone = std::make_shared<bool>(false);   // the directory is removed, whatever the fsync after it said
        return run_([this, dir, gone] {
                        if (env_.delete_hold_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(env_.delete_hold_ms));
                        kvflash_bundle_remove(dir);
                        std::error_code ec;
                        if (std::filesystem::exists(dir, ec)) return KvFlashResult{500, "cannot remove " + dir};
                        *gone = true;
                        if (!env_.fsync_dir(std::filesystem::path(dir).parent_path().string())) return KvFlashResult{500, "fsync of the sessions directory failed after the delete: it may reappear after a power loss"};
                        return KvFlashResult{};
                    },
                    [this, key, gone, then](const KvFlashResult & removed) {
                        KvFlashResult r = removed;
                        if (*gone) { r = delete_commit_(key); if (!removed.ok()) r = removed; }
                        note_op_(key, "delete", r.status >= 300 ? "failed" : "done", r.error);
                        return then(r);
                    }, /*continuation=*/true,
                    [this, key, then](const KvFlashResult & r) { note_op_(key, "delete", "failed", r.error); then(r); });
    }
    // The owner's part of a delete whose directory is gone: the entry, the decision and the unlink debt go with it.
    // ponytail: the meta.json/index.json write (and its fsync) of a delete or a compaction still runs here on the owner; moving it to
    // the worker needs the store guarded against a concurrent reader, which it is not.
    KvFlashResult delete_commit_(const std::string & key) {
        store_.erase(key);
        drop_pending_(key);
        { std::lock_guard<std::mutex> lk(mu_); unlink_debt_.erase(key); }
        KvFlashResult r;
        if (!store_.write_index()) r.error = "index.json not written (the scan rebuilds it)";
        return r;
    }
    void note_op_(const std::string & key, const char * op, const char * state, const std::string & err) {
        std::lock_guard<std::mutex> lk(mu_);
        ops_[key] = {op, state, err};
    }

    bool save_pending_() {
        std::string out = "{\"next_id\":" + std::to_string(next_id_) + "}\n";
        for (const auto & p : pending_) {
            out += "{\"id\":"; detail::json_str(out, p.id);
            out += ",\"key\":"; detail::json_str(out, p.key);
            out += ",\"reason\":"; detail::json_str(out, p.reason);
            out += ",\"state\":"; detail::json_str(out, p.state);
            out += ",\"gen\":" + std::to_string(p.gen) + ",\"last_used\":" + std::to_string(p.last_used) + ",\"created\":" + std::to_string(p.created) +
                   ",\"deadline\":" + std::to_string(p.deadline) + ",\"options\":\"";
            for (size_t i = 0; i < p.options.size(); ++i) out += (i ? "," : "") + p.options[i];
            out += "\"}\n";
        }
        return kvflash_bundle_write(store_.root() + "/pending.json", std::vector<uint8_t>(out.begin(), out.end())) && kvflash_fsync_dir(store_.root());
    }
    void drop_pending_(const std::string & key) {
        const size_t n = pending_.size();
        pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [&](const auto & p) { return p.key == key; }), pending_.end());
        if (pending_.size() != n) save_pending_();
    }
    void add_pending_(const KvFlashSessionEntry & e, const std::string & reason, int64_t now) {
        KvFlashPending p;
        p.id = "p" + std::to_string(next_id_++); p.key = e.key; p.reason = reason; p.state = e.state; p.gen = e.gen; p.last_used = e.last_used; p.created = now;
        p.deadline = now + (env_.grace_s >= 0 ? env_.grace_s : (int64_t)policy_.grace_hours * 3600);
        p.options = e.state == "compact" ? std::vector<std::string>{"export", "delete"} : std::vector<std::string>{"compact", "export", "delete"};
        pending_.push_back(std::move(p));
        save_pending_();
    }

    // The owner's part of a compaction, with the descriptor and the unlinks already done (or crashed, in a test).
    KvFlashResult compact_commit_(const std::string & key, const KvFlashSlotHead & h, const KvFlashCompactResult & cr) {
        const KvFlashSessionEntry * e = store_.find_key(key);
        if (!e) return {404, "no such session"};
        const std::string dir = dir_of_(key);
        if (cr.crashed) return {500, "crash injected"};
        if (!cr.committed) return {500, "compact.json not written for " + key};
        KvFlashSessionEntry u = *e;
        u.state = "compact"; u.bytes_live = kvflash_tree_bytes(dir); u.bytes_dead = 0; u.history.clear(); u.history.shrink_to_fit(); u.gen = h.gen;
        store_.upsert(u);
        const bool meta_ok = store_.write_meta(key);
        u.bytes_live = kvflash_tree_bytes(dir);   // meta.json changed size with its state; the file keeps the figure it was written with
        store_.upsert(u);
        if (env_.crash_at == (int)(3 + cr.unlinks)) return {500, "crash injected"};
        const bool index_ok = store_.write_index();
        drop_pending_(key);
        KvFlashResult r;
        if (!cr.left.empty()) {
            std::lock_guard<std::mutex> lk(mu_);
            unlink_debt_[key] = h.gen;
            r.error = "unlink failed:";
            for (const auto & f : cr.left) r.error += " " + f;
        }
        // the descriptor is the commit: a meta or index that did not land is repaired by the scan, but it is reported
        if (!meta_ok) { r.status = 500; r.error += (r.error.empty() ? "" : "; ") + std::string("meta.json of ") + key + " not rewritten (the descriptor is committed; the scan repairs it)"; }
        else if (!index_ok) r.error += (r.error.empty() ? "" : "; ") + std::string("index.json not written (the scan rebuilds it)");
        return r;
    }
    // A failed unlink is owed against one incarnation. The retry needs the entry to be that incarnation (compact, the generation
    // the debt was taken at), its head to load and a valid descriptor of that head; it removes only the files the descriptor lists.
    // The unlinks run on a worker under the key's claim; the owner settles the debt and the byte count at the next pump.
    void retry_unlinks_(const std::vector<KvFlashWork> & picked) {
        std::map<std::string, uint32_t> debt;
        { std::lock_guard<std::mutex> lk(mu_); debt = unlink_debt_; }
        for (const auto & [key, gen] : debt) {
            const KvFlashSessionEntry * e = store_.find_key(key);
            if (!e || e->state != "compact" || e->gen != gen) { std::lock_guard<std::mutex> lk(mu_); unlink_debt_.erase(key); continue; }
            if (is_held_(key) || std::any_of(picked.begin(), picked.end(), [&](const KvFlashWork & w) { return w.key == key; })) continue;
            const std::string dir = dir_of_(key);
            KvFlashSlotHead h;
            KvFlashCompactDescriptor d;
            if (!detail::read_head(dir, identity_, head_max_, h) || !kvflash_compact_validate(dir, h, d) || d.gen != gen) continue;
            if (!take(claim_of(key)).ok()) continue;
            auto left = std::make_shared<bool>(false);
            run_([this, dir, files = d.unlink, left] { for (const std::string & f : files) *left |= !env_.unlink_file(dir + "/" + f); return KvFlashResult{}; },
                 [this, key, dir, left](const KvFlashResult &) {
                     if (!*left) { std::lock_guard<std::mutex> lk(mu_); unlink_debt_.erase(key); }
                     for (auto & en : store_.entries()) if (en.key == key && en.state == "compact") en.bytes_live = kvflash_tree_bytes(dir);
                     release(key);
                     return KvFlashResult{};
                 }, /*continuation=*/false, [this, key](const KvFlashResult &) { release(key); });
        }
    }

    // Disk sessions whose log carries dead records are rewritten (copy-only) once idle, under a reserved peak: the plan is
    // made here, the copy runs on a worker, the owner commits the result at the next pump. One rewrite at a time.
    // ponytail: one log rewrite in flight keeps the reserved peaks small; a queue of them comes if a store of many dead logs shows up
    // No room for the duplicate log: the compact tier when the policy compacts, a "space" decision when it asks, else left.
    void reclaim_logs_(int64_t now) {
        if (policy_.log_dead_mib == 0) return;
        std::vector<std::string> keys;
        for (const auto & e : store_.entries())
            if (e.state == "disk" && e.bytes_dead >= policy_.log_dead_mib * env_.mib && now - e.last_used >= 30 && !is_held_(e.key)) keys.push_back(e.key);
        for (const std::string & key : keys) {
            if (!log_inflight_.empty()) break;
            const KvFlashClaim claim = claim_of(key);
            const std::string dir = dir_of_(key);
            KvFlashSlotHead h;
            if (!detail::read_head(dir, identity_, head_max_, h)) continue;
            KvFlashSlotHead nh;
            const KvFlashBundleCommit c = kvflash_log_compaction_commit(dir, h, rb_, nh);
            KvFlashSessionEntry u = *store_.find_key(key);
            u.gen = nh.gen; u.bytes_dead = 0;
            uint64_t res = 0;
            std::string why;
            if (!reserve_commit(c, u, res, why)) {
                if (policy_.action == "compact") compact(claim);
                else if (policy_.action == "ask") ask_space_(store_.find_key(key), now);
                continue;
            }
            if (!take(claim).ok()) { release_reservation(res); continue; }
            log_inflight_.insert(key);
            run_([c] { return kvflash_bundle_commit(c) ? KvFlashResult{} : KvFlashResult{500, "log rewrite not committed"}; },
                 [this, key, res, u](const KvFlashResult & done) mutable {
                     release_reservation(res);
                     log_inflight_.erase(key);
                     if (done.ok() && store_.find_key(key)) {
                         u.bytes_live = kvflash_tree_bytes(dir_of_(key));
                         store_.upsert(u);
                         if (!store_.write_meta(key) || !store_.write_index()) std::fprintf(stderr, "[kvflash] storage: meta or index of %s not written after its log rewrite\n", key.c_str());
                     } else if (!done.ok()) std::fprintf(stderr, "[kvflash] storage: log rewrite of %s failed: %s\n", key.c_str(), done.error.c_str());
                     release(key);
                     return KvFlashResult{};
                 }, false,
                 [this, key, res](const KvFlashResult &) { release_reservation(res); log_inflight_.erase(key); release(key); });
        }
    }

    // What a copy must satisfy: the head loads and the park it names would restore here (the scan's validator: the log header,
    // every record's tag and crc, the ledger, each slab walked).
    bool verify_bundle_dir_(const std::string & dir, std::string & why) const {
        KvFlashSlotHead h;
        if (!detail::read_head(dir, identity_, head_max_, h)) { why = "head does not load"; return false; }
        return kvflash_bundle_validate(dir, h, rb_ - kKvFlashRecordTag, env_.slab_max, why);
    }

    KvFlashSessionStore & store_;
    uint64_t identity_;
    size_t rb_, head_max_;
    KvFlashStorageEnv env_;
    KvFlashPolicy policy_;
    std::vector<KvFlashPending> pending_;
    uint64_t next_id_ = 1;
    int64_t tick_at_ = std::numeric_limits<int64_t>::min() / 2;   // the owner thread's last steward pass
    mutable std::mutex mu_;
    std::set<std::string> busy_;
    std::map<std::string, uint32_t> unlink_debt_;   // key -> the head generation of the compaction that still owes unlinks
    std::map<std::string, int> writers_;
    std::vector<Ack> acks_;
    std::map<uint64_t, uint64_t> reserved_;
    uint64_t next_res_ = 0;
    bool stopping_ = false;   // under mu_: shutdown() was called, no work starts
    std::map<std::string, ExportRecord> exports_, imports_, ops_;   // ops_: path is the operation (compact | delete), state running | done | failed   // imports_: path unused, state importing | imported | failed
    std::vector<std::function<void()>> finished_;
    std::vector<Worker> workers_;
    std::set<std::string> log_inflight_;   // owner thread only: logs being rewritten
    std::mutex wake_mu_;
    std::function<void()> wake_fn_;
};

// ---- what the engine calls --------------------------------------------------------------------

// The engine's decision before it hands a park to the writer: the commit's real bytes are reserved; a refusal skips
// the park and marks the slot to rewrite from its acknowledged head on the next one (the session stays in its slot,
// so it is not evicted). No storage: nothing to ask.
inline bool kvflash_park_admit(KvFlashStorage * st, const KvFlashBundleCommit & c, const KvFlashSessionEntry & e, KvFlashSlotBundle & sb, uint64_t & res, std::string & why) {
    res = 0;
    if (!st || st->reserve_commit_with_room(c, e, res, why)) return true;
    sb.rewrite = true;
    return false;
}

// What KvFlashSessionBook::resume_claim is wired to: a restore takes the key's storage claim first (a decision about it
// is void; an action in flight denies it, and the book answers busy so the request retries instead of admitting cold).
inline std::function<bool(const std::string &)> kvflash_resume_claim(KvFlashStorage * st) {
    if (!st) return {};
    return [st](const std::string & key) { return st->claim_resume(key).ok(); };
}

// The engine's one wiring of the storage into the book: a restore takes the key's storage claim (see kvflash_resume_claim).
inline void kvflash_attach_storage(KvFlashSessionBook & book, KvFlashStorage * st) { book.resume_claim = kvflash_resume_claim(st); }

}  // namespace luce::common
