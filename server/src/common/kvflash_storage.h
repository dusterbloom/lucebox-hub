// KVFlash 2.0 sessions, phase P3 (trackc/SESSIONS_DESIGN.md section 5): the storage policy over the
// session store. Policy (quota, age, action, grace, export path) persisted in policy.json; the
// tier state machine live -> disk -> compact -> exported/gone; the ask flow (a pending decision at
// quota or age, compact after the grace, delete only on an explicit yes); generation claims so a
// click, a timeout, a resume, an export and a queued writer never act on the same instance twice;
// occupancy from stat with the peak of every write reserved first; export and import.
//
// Threading: every call that reads or changes the store runs on one owner thread (the engine's,
// through the scheduler command path); the claim tables, reservations and the finished-export queue
// are under mu_ because an export's copy runs on a worker that touches only its own plan.
#pragma once

#include "kvflash_session_store.h"

#include <algorithm>
#include <cerrno>
#include <ctime>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>

namespace luce::common {

// ---- the policy -------------------------------------------------------------------------------

struct KvFlashPolicy {
    uint64_t quota_gib = 64;              // 0: free space minus the reserve
    int64_t reserve_gib = -1;             // -1 ("auto"): max(10% of the disk, 20 GiB)
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
inline bool parse_u64(const std::string & s, uint64_t & out) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || s.size() > 18) return false;
    out = std::strtoull(s.c_str(), nullptr, 10);
    return true;
}
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
}  // namespace detail

// A partial policy object merged over `p`; all or nothing. False with `err` naming the field.
inline bool kvflash_policy_apply(std::string_view json, KvFlashPolicy & p, std::string & err) {
    std::map<std::string, std::string> m;
    if (!detail::json_flat_object(json, m)) { err = "body: not a flat JSON object"; return false; }
    KvFlashPolicy q = p;
    auto num = [&](const char * k, uint64_t & dst) {
        const auto it = m.find(k);
        if (it == m.end()) return true;
        if (!detail::parse_u64(it->second, dst)) { err = std::string(k) + ": a non-negative integer is required"; return false; }
        return true;
    };
    auto str = [&](const char * k, std::string & dst, std::initializer_list<const char *> allowed) {
        const auto it = m.find(k);
        if (it == m.end()) return true;
        if (allowed.size() && std::none_of(allowed.begin(), allowed.end(), [&](const char * a) { return it->second == a; })) {
            err = std::string(k) + ": must be one of";
            for (const char * a : allowed) err += std::string(" ") + a;
            return false;
        }
        dst = it->second;
        return true;
    };
    for (const auto & kv : m)
        if (!(kv.first == "quota_gib" || kv.first == "reserve_gib" || kv.first == "max_age_days" || kv.first == "action" || kv.first == "grace_hours" ||
              kv.first == "export_path" || kv.first == "after_export" || kv.first == "log_dead_mib" || kv.first == "preamble_mib")) {
            err = kv.first + ": unknown field";
            return false;
        }
    if (!str("action", q.action, {"ask", "compact", "export", "delete-oldest"}) || !num("quota_gib", q.quota_gib) || !num("max_age_days", q.max_age_days) ||
        !num("grace_hours", q.grace_hours) || !str("after_export", q.after_export, {"delete", "keep"}) || !str("export_path", q.export_path, {}) ||
        !num("log_dead_mib", q.log_dead_mib) || !num("preamble_mib", q.preamble_mib)) return false;
    if (const auto it = m.find("reserve_gib"); it != m.end()) {
        uint64_t v = 0;
        if (it->second == "auto") q.reserve_gib = -1;
        else if (detail::parse_u64(it->second, v)) q.reserve_gib = (int64_t)v;
        else { err = "reserve_gib: \"auto\" or a non-negative integer is required"; return false; }
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

// ---- the storage ------------------------------------------------------------------------------

struct KvFlashStorageEnv {
    std::function<int64_t()> now;                          // unix seconds
    std::function<uint64_t()> free_bytes, disk_bytes;      // the filesystem under the root
    std::function<bool(const std::string &)> unlink_file;  // false: the unlink failed
    uint64_t gib = 1ull << 30, mib = 1ull << 20;           // the units of the policy (tests shrink them)
    int64_t grace_s = -1;                                  // >= 0 replaces grace_hours (LUCE_KVFLASH_GRACE_S, a test-only knob)
    int crash_at = 0;                                      // the compact transition's crash point (tests)
    bool async_export = false;                             // the copy runs on a worker thread
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

struct KvFlashOccupancy { uint64_t live = 0, dead = 0, total = 0, reserved_peaks = 0, quota = 0, free = 0; };

struct KvFlashExportPlan {
    KvFlashClaim claim;
    std::string key, src, tmp, dst;
    std::vector<std::string> files;
    bool compact = false, delete_local = false;
    uint64_t bytes = 0, reservation = 0;
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
    }
    ~KvFlashStorage() { for (auto & t : workers_) if (t.joinable()) t.join(); }
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
    // A park of `key` is queued on the writer (its slot may be reused meanwhile) and is acknowledged later.
    void writer_begin(const std::string & key) { std::lock_guard<std::mutex> lk(mu_); ++writers_[key]; }
    void writer_ack(const std::string & key) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = writers_.find(key);
        if (it != writers_.end() && --it->second <= 0) writers_.erase(it);
    }

    // ---- compact and delete ----

    KvFlashResult compact(const KvFlashClaim & c) {
        KvFlashResult t = take(c);
        if (!t.ok()) return t;
        KvFlashResult r = compact_claimed_(c.key);
        release(c.key);
        return r;
    }
    KvFlashResult remove(const KvFlashClaim & c) {
        KvFlashResult t = take(c);
        if (!t.ok()) return t;
        KvFlashResult r;
        const KvFlashSessionEntry * e = store_.find_key(c.key);
        if (kvflash_state_next(e->state, "delete").empty()) r = {409, "the session is in a slot (state " + e->state + ")"};
        else {
            const std::string dir = dir_of_(c.key);
            kvflash_bundle_remove(dir);
            kvflash_fsync_dir(std::filesystem::path(dir).parent_path().string());
            store_.erase(c.key);
            drop_pending_(c.key);
            store_.write_index();
        }
        release(c.key);
        return r;
    }

    // ---- export and import ----

    KvFlashResult export_begin(const KvFlashClaim & c, const std::string & path_arg, KvFlashExportPlan & plan) {
        const std::string path = path_arg.empty() ? policy_.export_path : path_arg;
        if (path.empty()) return {400, "no export path: set export_path in the policy or pass path"};
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) return {400, "export path " + path + " does not exist"};
        if (!std::filesystem::is_directory(path, ec)) return {400, "export path " + path + " is not a directory"};
        if (::access(path.c_str(), W_OK | X_OK) != 0) return {400, "export path " + path + " is not writable"};
        KvFlashResult t = take(c);
        if (!t.ok()) return t;
        auto fail = [&](KvFlashResult r) { release(c.key); return r; };
        const KvFlashSessionEntry * e = store_.find_key(c.key);
        if (kvflash_state_next(e->state, "export").empty()) return fail({409, "the session cannot be exported in state " + e->state});
        KvFlashExportPlan p;
        p.claim = c; p.key = c.key; p.src = dir_of_(c.key);
        p.dst = path + "/" + c.key; p.tmp = p.dst + ".tmp";
        if (std::filesystem::exists(p.dst, ec)) return fail({409, "export destination " + p.dst + " exists"});
        p.compact = e->state == "compact";
        p.files = {"head", "meta.json"};
        if (p.compact) p.files.push_back(kCompactFile);
        else {
            KvFlashSlotHead h;
            if (!detail::read_head(p.src, identity_, head_max_, h)) return fail({500, "head of " + c.key + " does not load"});
            p.files.push_back(kvflash_bundle_log_name(h.log_gen));
            for (uint32_t id : {h.end_gen, h.ckpt_gen, h.trunk_gen}) if (id) p.files.push_back(kvflash_bundle_slab_name(id));
        }
        for (const auto & f : p.files) p.bytes += detail::file_size_or0(p.src + "/" + f);
        p.delete_local = policy_.after_export == "delete";
        struct stat a{}, b{};
        if (::stat(p.src.c_str(), &a) == 0 && ::stat(path.c_str(), &b) == 0 && a.st_dev == b.st_dev) {   // same filesystem: the copy needs the session's bytes again
            std::string why;
            if (!reserve(p.bytes, p.reservation, why)) return fail({507, why});
        }
        plan = std::move(p);
        return {};
    }
    // The copy: no store access, safe on a worker. <key>.tmp/ is written, every record's crc is verified
    // on the copy, files and directory are fsynced, then renamed to <key>/.
    KvFlashResult export_copy(const KvFlashExportPlan & p) const {
        std::error_code ec;
        std::filesystem::remove_all(p.tmp, ec);
        if (::mkdir(p.tmp.c_str(), 0755) != 0) return {errno == ENOSPC ? 507 : 500, "cannot create " + p.tmp + ": " + std::strerror(errno)};
        auto fail = [&](KvFlashResult r) { std::filesystem::remove_all(p.tmp, ec); return r; };
        for (const auto & f : p.files) {
            std::string why;
            if (!detail::copy_file_synced(p.src + "/" + f, p.tmp + "/" + f, why)) return fail({errno == ENOSPC ? 507 : 500, why});
        }
        if (!p.compact) {
            std::string why;
            if (!verify_bundle_dir_(p.tmp, why)) return fail({500, "the copy does not verify: " + why});
        }
        if (!kvflash_fsync_dir(p.tmp)) return fail({500, "fsync of " + p.tmp + " failed"});
        if (std::rename(p.tmp.c_str(), p.dst.c_str()) != 0) return fail({500, "rename to " + p.dst + ": " + std::strerror(errno)});
        kvflash_fsync_dir(std::filesystem::path(p.dst).parent_path().string());
        return {};
    }
    KvFlashResult export_commit(const KvFlashExportPlan & p, const KvFlashResult & copied) {
        if (p.reservation) release_reservation(p.reservation);
        ExportRecord rec{p.dst, copied.ok() ? (p.delete_local ? "exported" : "copied") : "failed", copied.error};
        KvFlashResult r = copied;
        if (copied.ok() && p.delete_local) {   // only after the verified copy
            kvflash_bundle_remove(p.src);
            kvflash_fsync_dir(std::filesystem::path(p.src).parent_path().string());
            store_.erase(p.key);
            drop_pending_(p.key);
            store_.write_index();
        }
        { std::lock_guard<std::mutex> lk(mu_); exports_[p.key] = rec; }
        release(p.key);
        return r;
    }
    KvFlashResult export_session(const KvFlashClaim & c, const std::string & path) {
        KvFlashExportPlan plan;
        KvFlashResult r = export_begin(c, path, plan);
        if (!r.ok()) return r;
        return export_commit(plan, export_copy(plan));
    }
    // 202 with the copy on a worker (committed by the owner's next tick) when env.async_export, else done.
    KvFlashResult export_async(const KvFlashClaim & c, const std::string & path) {
        if (!env_.async_export) return export_session(c, path);
        KvFlashExportPlan plan;
        KvFlashResult r = export_begin(c, path, plan);
        if (!r.ok()) return r;
        workers_.emplace_back([this, plan] {
            KvFlashResult copied = export_copy(plan);
            std::lock_guard<std::mutex> lk(mu_);
            finished_.push_back({plan, std::move(copied)});
        });
        return {202, ""};
    }
    void commit_finished_exports() {
        std::vector<Finished> done;
        { std::lock_guard<std::mutex> lk(mu_); done.swap(finished_); }
        for (auto & f : done) export_commit(f.plan, f.copied);
    }
    bool export_result(const std::string & key, std::string & path, std::string & state, std::string * err = nullptr) const {
        std::lock_guard<std::mutex> lk(mu_);
        const auto it = exports_.find(key);
        if (it == exports_.end()) return false;
        path = it->second.path; state = it->second.state;
        if (err) *err = it->second.error;
        return true;
    }

    // `path` is an exported session directory. Never overwrites: it lands under a key of its own.
    KvFlashResult import_dir(const std::string & path, bool as_foreign, std::string * key_out) {
        std::error_code ec;
        if (!std::filesystem::is_directory(path, ec)) return {400, path + " is not a directory"};
        if (!std::filesystem::exists(path + "/head", ec)) return {400, path + " holds no session (no head)"};
        std::vector<uint8_t> mb;
        KvFlashSessionEntry e;
        if (!kvflash_bundle_read(path + "/meta.json", mb, 1 << 20) || !kvflash_meta_unpack(std::string_view((const char *)mb.data(), mb.size()), e))
            return {400, path + " has no readable meta.json"};
        KvFlashSlotHead h;
        const bool ours = detail::read_head(path, identity_, head_max_, h);
        const bool compact = std::filesystem::exists(path + "/" + kCompactFile, ec);
        if (!ours && !as_foreign) {
            char b[96];
            std::snprintf(b, sizeof b, "identity mismatch: the head is of identity %016llx, this server is %016llx",
                          (unsigned long long)detail::peek_head_identity(path), (unsigned long long)identity_);
            return {409, b};
        }
        std::string why;
        if (ours && !compact && !verify_bundle_dir_(path, why)) return {400, "the bundle does not verify: " + why};
        std::vector<std::string> files;
        if (!ours) { for (const auto & de : std::filesystem::directory_iterator(path, ec)) if (de.is_regular_file(ec)) files.push_back(de.path().filename().string()); }
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
        uint64_t res = 0;
        if (!reserve(bytes, res, why)) return {507, why};
        const std::string key = store_.new_key(e.client, e.sid);
        const std::string tmp = store_.dir(key) + ".tmp";
        std::filesystem::remove_all(tmp, ec);
        KvFlashResult r;
        if (::mkdir(tmp.c_str(), 0755) != 0) r = {500, "cannot create " + tmp + ": " + std::strerror(errno)};
        for (size_t i = 0; r.ok() && i < files.size(); ++i)
            if (!detail::copy_file_synced(path + "/" + files[i], tmp + "/" + files[i], why)) r = {errno == ENOSPC ? 507 : 500, why};
        if (r.ok() && (!kvflash_fsync_dir(tmp) || std::rename(tmp.c_str(), store_.dir(key).c_str()) != 0)) r = {500, "cannot publish " + store_.dir(key)};
        release_reservation(res);
        if (!r.ok()) { std::filesystem::remove_all(tmp, ec); return r; }
        kvflash_fsync_dir(store_.dir(""));
        e.key = key; e.state = !ours ? "foreign" : compact ? "compact" : "disk"; e.identity = ours ? identity_ : detail::peek_head_identity(path);
        e.archive_of.clear(); e.archive.clear();
        if (ours) { e.tokens = h.history.size(); e.gen = h.gen; if (!compact) e.history = std::move(h.history); }
        e.bytes_live = kvflash_tree_bytes(store_.dir(key)); e.bytes_dead = 0;
        store_.upsert(e);
        store_.write_meta(key);
        store_.write_index();
        if (key_out) *key_out = key;
        return {};
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
        return policy_.reserve_gib >= 0 ? (uint64_t)policy_.reserve_gib * env_.gib : std::max<uint64_t>(env_.disk_bytes() / 10, 20 * env_.gib);
    }
    uint64_t quota_bytes(uint64_t total, uint64_t free) const {
        if (policy_.quota_gib > 0) return policy_.quota_gib * env_.gib;
        const uint64_t room = total + free, reserve = reserve_bytes();
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
        for (const auto & r : reserved_) o.reserved_peaks += r.second;
        return o;
    }
    // Reserve the peak of a write before it is scheduled: free - reserve - reserved peaks >= peak, and
    // no growth past 1.1 x quota. A refusal means the write is not scheduled (a park is skipped, the session stays in its slot).
    bool reserve(uint64_t peak, uint64_t & id, std::string & why) {
        const uint64_t free = env_.free_bytes(), floor = reserve_bytes();
        if (policy_.quota_gib > 0 && peak > 0) {
            const uint64_t total = kvflash_tree_bytes(store_.root());
            if (total > policy_.quota_gib * env_.gib / 10 * 11) { why = "the store is past 1.1 x quota: new durable growth is refused"; return false; }
        }
        std::lock_guard<std::mutex> lk(mu_);
        uint64_t held = 0;
        for (const auto & r : reserved_) held += r.second;
        if (free < floor + held + peak) {
            why = "free " + std::to_string(free) + " is below the reserve " + std::to_string(floor) + " + reserved peaks " + std::to_string(held) + " + this write's peak " + std::to_string(peak);
            return false;
        }
        id = ++next_res_;
        reserved_[id] = peak;
        return true;
    }
    void release_reservation(uint64_t id) { std::lock_guard<std::mutex> lk(mu_); reserved_.erase(id); }

    // ---- the steward ----

    // One pass: finish exports, retry unlinks the descriptor lists, drop stale decisions, compact what passed
    // its grace, rewrite dead logs, then act on quota and age by the policy. Returns the exports it wants started.
    std::vector<KvFlashWork> tick() {
        std::vector<KvFlashWork> work;
        commit_finished_exports();
        const int64_t now = env_.now();
        retry_unlinks_();
        // decisions whose instance moved on are void; an expired one compacts, never deletes
        for (const KvFlashPending & p : pending()) {
            const KvFlashSessionEntry * e = store_.find_key(p.key);
            if (!e) { drop_pending_(p.key); continue; }
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
                if (compact(claim).ok()) need -= std::max<int64_t>(0, bytes - (int64_t)kvflash_tree_bytes(dir_of_(e.key)));
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
        return work;
    }

private:
    struct ExportRecord { std::string path, state, error; };
    struct Finished { KvFlashExportPlan plan; KvFlashResult copied; };

    bool is_held_(const std::string & key) const { std::lock_guard<std::mutex> lk(mu_); auto w = writers_.find(key); return busy_.count(key) || (w != writers_.end() && w->second > 0); }
    bool has_pending_(const std::string & key) const { for (const auto & p : pending_) if (p.key == key) return true; return false; }
    static bool has_option_(const KvFlashPending & p, const char * o) { return std::find(p.options.begin(), p.options.end(), o) != p.options.end(); }
    // an old instance whose successor is live is the live session's archive: not aged out from under it
    bool exempt_(const KvFlashSessionEntry & e) const {
        if (e.archive.empty()) return false;
        const KvFlashSessionEntry * s = store_.find_key(e.archive);
        return s && s->state == "parked-gpu";
    }
    std::string dir_of_(const std::string & key) const {   // a foreign slot<i>/ the migration left in place lives at the root
        return detail::is_slot_dir_name(key) ? store_.root() + "/" + key : store_.dir(key);
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

    KvFlashResult compact_claimed_(const std::string & key) {
        const KvFlashSessionEntry * e = store_.find_key(key);
        if (kvflash_state_next(e->state, "compact").empty()) return {409, "a session in state " + e->state + " cannot be compacted"};
        const std::string dir = dir_of_(key);
        KvFlashSlotHead h;
        if (!detail::read_head(dir, identity_, head_max_, h)) return {500, "head of " + key + " does not load"};
        const KvFlashCompactResult cr = kvflash_compact_dir(dir, h, env_.unlink_file, env_.crash_at);
        if (cr.crashed) return {500, "crash injected"};
        if (!cr.committed) return {500, "compact.json not written for " + key};
        KvFlashSessionEntry u = *e;
        u.state = "compact"; u.bytes_live = kvflash_tree_bytes(dir); u.bytes_dead = 0; u.history.clear(); u.history.shrink_to_fit(); u.gen = h.gen;
        store_.upsert(u);
        store_.write_meta(key);
        u.bytes_live = kvflash_tree_bytes(dir);   // meta.json changed size with its state; the file keeps the figure it was written with
        store_.upsert(u);
        if (env_.crash_at == (int)(3 + cr.unlinks)) return {500, "crash injected"};
        store_.write_index();
        drop_pending_(key);
        KvFlashResult r;
        if (!cr.left.empty()) {
            std::lock_guard<std::mutex> lk(mu_);
            unlink_debt_.insert(key);
            r.error = "unlink failed:";
            for (const auto & f : cr.left) r.error += " " + f;
        }
        return r;
    }
    void retry_unlinks_() {
        std::set<std::string> debt;
        { std::lock_guard<std::mutex> lk(mu_); debt = unlink_debt_; }
        for (const std::string & key : debt) {
            if (is_held_(key)) continue;
            bool left = false;
            for (const auto & f : kvflash_compact_leftovers(dir_of_(key))) left |= !env_.unlink_file(dir_of_(key) + "/" + f);
            if (left) continue;
            std::lock_guard<std::mutex> lk(mu_);
            unlink_debt_.erase(key);
        }
        for (auto & e : store_.entries()) if (debt.count(e.key) && e.state == "compact") e.bytes_live = kvflash_tree_bytes(dir_of_(e.key));
    }

    // Disk sessions whose log carries dead records are rewritten (copy-only) once idle, under a reserved peak. No room
    // for the duplicate log: the compact tier when the policy compacts, a "space" decision when it asks, else left.
    void reclaim_logs_(int64_t now) {
        if (policy_.log_dead_mib == 0) return;
        std::vector<std::string> keys;
        for (const auto & e : store_.entries())
            if (e.state == "disk" && e.bytes_dead >= policy_.log_dead_mib * env_.mib && now - e.last_used >= 30 && !is_held_(e.key)) keys.push_back(e.key);
        for (const std::string & key : keys) {
            const KvFlashClaim claim = claim_of(key);
            const std::string dir = dir_of_(key);
            KvFlashSlotHead h;
            if (!detail::read_head(dir, identity_, head_max_, h)) continue;
            uint64_t live = 0;
            for (uint64_t o : h.map) live += o != 0;
            uint64_t res = 0;
            std::string why;
            if (!reserve(live * rb_ + kKvFlashLogHeader, res, why)) {
                if (policy_.action == "compact") compact(claim);
                else if (policy_.action == "ask" && !has_pending_(key)) add_pending_(*store_.find_key(key), "space", now);
                continue;
            }
            if (take(claim).ok()) {
                KvFlashSlotHead nh;
                const KvFlashBundleCommit c = kvflash_log_compaction_commit(dir, h, rb_, nh);
                if (kvflash_bundle_commit(c)) {
                    KvFlashSessionEntry u = *store_.find_key(key);
                    u.gen = nh.gen; u.bytes_dead = 0; u.bytes_live = kvflash_tree_bytes(dir);
                    store_.upsert(u);
                    store_.write_meta(key);
                    store_.write_index();
                }
                release(key);
            }
            release_reservation(res);
        }
    }

    // The log's records read back from `dir` through their crc, a slab file per named id: what a copy must satisfy.
    bool verify_bundle_dir_(const std::string & dir, std::string & why) const {
        KvFlashSlotHead h;
        if (!detail::read_head(dir, identity_, head_max_, h)) { why = "head does not load"; return false; }
        const std::string log = dir + "/" + kvflash_bundle_log_name(h.log_gen);
        FILE * f = std::fopen(log.c_str(), "rb");
        if (!f) { why = "no " + kvflash_bundle_log_name(h.log_gen); return false; }
        std::vector<uint8_t> rec(rb_);
        bool ok = true;
        uint8_t lh[kKvFlashLogHeader];
        uint32_t gen = 0;
        ok = std::fread(lh, 1, sizeof lh, f) == sizeof lh && kvflash_log_header_parse(lh, sizeof lh, gen) && gen == h.log_gen;
        if (!ok) why = "log header";
        for (size_t c = 0; ok && c < h.map.size(); ++c) {
            if (h.map[c] == 0) continue;
            KvFlashRecordTag t;
            ok = h.map[c] >= kKvFlashLogHeader && h.map[c] + rb_ <= h.log_bytes && ::fseeko(f, (off_t)h.map[c], SEEK_SET) == 0 && std::fread(rec.data(), 1, rb_, f) == rb_;
            if (ok) { kvflash_record_tag_parse(rec.data(), rb_, t); ok = t.chunk == c && t.crc == kvflash_crc64(rec.data() + kKvFlashRecordTag, rb_ - kKvFlashRecordTag); }
            if (!ok) why = "record of chunk " + std::to_string(c) + " (short, wrong chunk or crc)";
        }
        std::fclose(f);
        for (uint32_t id : {h.end_gen, h.ckpt_gen, h.trunk_gen})
            if (ok && id && !std::filesystem::exists(dir + "/" + kvflash_bundle_slab_name(id))) { ok = false; why = "missing " + kvflash_bundle_slab_name(id); }
        return ok;
    }

    KvFlashSessionStore & store_;
    uint64_t identity_;
    size_t rb_, head_max_;
    KvFlashStorageEnv env_;
    KvFlashPolicy policy_;
    std::vector<KvFlashPending> pending_;
    uint64_t next_id_ = 1;
    mutable std::mutex mu_;
    std::set<std::string> busy_, unlink_debt_;
    std::map<std::string, int> writers_;
    std::map<uint64_t, uint64_t> reserved_;
    uint64_t next_res_ = 0;
    std::map<std::string, ExportRecord> exports_;
    std::vector<Finished> finished_;
    std::vector<std::thread> workers_;
};

}  // namespace luce::common
