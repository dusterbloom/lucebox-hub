// KVFlash 2.0 sessions, phase P0 (trackc/SESSIONS_DESIGN.md sections 2 and 5.5): the session
// store. A session is a directory <dir>/sessions/<key>/ holding one parked conversation's bundle
// (head, chunks.<g>.log, slab.<id>: the kvflash_slot_bundle.h formats, unchanged) plus meta.json.
// The N engine slots are a cache over it: a full house moves a session to the disk tier instead of
// deleting it, and a request whose sid names a disk session loads it back.
//
//   identity    request headers (lower-cased map) + body session_id -> {client, sid}; a request
//               that sends nothing is anonymous (ownerless: resumable by content alone)
//   layout      <dir>/index.json (a cache), <dir>/sessions/<key>/{meta.json, head, ...}
//   scan        rebuilds the index from the directories: the head is the truth, meta.json says who
//   migration   today's <dir>/slot<i>/ renames into sessions/<key>/ (idempotent)
//
// Pure: files and strings only, no engine; every decision is testable on a directory.
#pragma once

#include "kvflash_slot_bundle.h"
#include "kvflash_slot_turn.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <vector>

namespace luce::common {

// ---- identity ---------------------------------------------------------------------------------

using KvFlashHeaders = std::map<std::string, std::string>;   // names lower-cased, values trimmed

constexpr size_t kKvFlashMetaMax = 1 << 20;   // what a reader takes of a meta.json
constexpr size_t kKvFlashSidMax = 1024;       // bytes of a session id: its meta.json (6x with escapes) stays far inside kKvFlashMetaMax

namespace detail {
inline std::string_view trim_ws(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.remove_suffix(1);
    return s;
}
}  // namespace detail

// The header lines of a request (after the request line). A name is stored lower-cased; a
// duplicate keeps its first value; a line without a colon is ignored.
inline KvFlashHeaders kvflash_parse_headers(std::string_view block) {
    KvFlashHeaders out;
    while (!block.empty()) {
        const size_t nl = block.find('\n');
        std::string_view line = block.substr(0, nl);
        block = nl == std::string_view::npos ? std::string_view{} : block.substr(nl + 1);
        const size_t colon = line.find(':');
        if (colon == std::string_view::npos) continue;
        std::string name(detail::trim_ws(line.substr(0, colon)));
        if (name.empty()) continue;
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        out.emplace(std::move(name), std::string(detail::trim_ws(line.substr(colon + 1))));
    }
    return out;
}

struct KvFlashSessionId {
    std::string sid, client;   // client: opencode | claude | header | body; both empty: no identity
    bool empty() const { return sid.empty(); }
};

// First hit wins: x-opencode-session-id, x-session-affinity (opencode), X-Claude-Code-Session-Id,
// X-Session-Id, then the body's session_id. Parent ids are a relation, never an identity.
inline KvFlashSessionId kvflash_resolve_session(const KvFlashHeaders & headers, const std::string & body_sid) {
    auto header = [&](const char * name) -> const std::string * {
        const auto it = headers.find(name);
        return it != headers.end() && !it->second.empty() ? &it->second : nullptr;
    };
    if (const std::string * v = header("x-opencode-session-id")) return {*v, "opencode"};
    if (const std::string * v = header("x-session-affinity")) return {*v, "opencode"};
    if (const std::string * v = header("x-claude-code-session-id")) return {*v, "claude"};
    if (const std::string * v = header("x-session-id")) return {*v, "header"};
    if (!body_sid.empty()) return {body_sid, "body"};
    return {};
}

// An identity the store can write down: its meta.json must fit what a reader takes (the HTTP boundary answers 400 otherwise).
inline bool kvflash_identity_valid(const KvFlashSessionId & id) { return id.sid.size() <= kKvFlashSidMax; }

// A client that named its session, as against one that sent nothing (anon) or a session migrated
// from slot<i>/ (its sid is made up): those two are ownerless, resumable by content alone.
inline bool kvflash_client_explicit(const std::string & client) {
    return client == "opencode" || client == "claude" || client == "header" || client == "body";
}

inline std::string kvflash_hex16(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)v);
    return buf;
}

// A session's identity is the pair: the same sid under two clients is two sessions.
inline std::string kvflash_sid_key(const std::string & client, const std::string & sid) { return kvflash_hex16(kvflash_fnv1a(client + "/" + sid)); }

// The directory key of a new session instance: fnv(client/sid), or, when a parked instance already
// owns that directory (a conversation that stopped matching its parked self), the first free
// fnv(client/sid#n). Nothing is ever overwritten.
inline std::string kvflash_session_dir_key(const std::string & client, const std::string & sid, const std::set<std::string> & taken) {
    std::string key = kvflash_sid_key(client, sid);
    for (int n = 1; taken.count(key); ++n) key = kvflash_sid_key(client, sid + "#" + std::to_string(n));
    return key;
}

// A parked session and who owns it, for the resume pick.
struct KvFlashOwnedSession {
    KvFlashParkedSession parked;
    std::string client, sid;
};

// The parked session a request resumes. An explicit request takes only its own (client, sid), from
// the first shared token; failing that, an ownerless session its prompt extends beyond the shared
// preamble (lcp > e_role). An anonymous request takes ownerless sessions only, under the same bar.
// Content never overrides a stated identity.
inline KvFlashSessionPick kvflash_pick_parked_owner(const std::vector<KvFlashOwnedSession> & parked, const std::string & req_client,
                                                    const std::string & req_sid, const std::vector<int32_t> & prompt, int e_role) {
    const bool req_explicit = kvflash_client_explicit(req_client);
    std::vector<KvFlashParkedSession> own, free;
    for (const auto & o : parked) {
        if (kvflash_client_explicit(o.client)) { if (req_explicit && o.client == req_client && o.sid == req_sid) own.push_back(o.parked); }
        else free.push_back(o.parked);
    }
    KvFlashSessionPick p = kvflash_pick_session(own, prompt, 1);
    return p.exact() ? p : kvflash_pick_session(free, prompt, std::max(e_role, 0) + 1);
}

// A resume that rewound its parked session to the resume point L (slab restored, pager truncated, history cut to L) and was then
// refused busy leaves the session valid at L but shorter than the prompt that picked it: the retry's pick/plan would see lcp <= L
// and reject the resume (L <= e_role) or cold-admit over the waiting session. The wait remembers that prompt's prefix, so the
// retry picks the same session at the same lcp.
struct KvFlashResumeWait {
    int L = 0, lcp = 0;               // the rewind point and the original shared prefix (L == 0: no wait)
    uint64_t prefix_hash = 0;         // of the original prompt's first lcp tokens
    std::string client, sid;          // who waits
};
inline uint64_t kvflash_prefix_hash(const std::vector<int32_t> & prompt, int n) {
    return kvflash_fnv1a(std::string_view((const char *)prompt.data(), (size_t)std::max(n, 0) * 4));
}
inline KvFlashResumeWait kvflash_resume_wait_make(int L, int lcp, const std::vector<int32_t> & prompt, const std::string & client, const std::string & sid) {
    return {L, lcp, kvflash_prefix_hash(prompt, lcp), client, sid};
}
// The pick for a request when `slot` holds a resume wait (history_size tokens parked): the same session at the original lcp when this
// prompt starts with the waiting prompt's prefix and comes from the same identity, else `pick` unchanged.
inline KvFlashSessionPick kvflash_pick_with_wait(const KvFlashSessionPick & pick, int slot, int history_size, const KvFlashResumeWait & wait,
                                                 const std::vector<int32_t> & prompt, const std::string & client, const std::string & sid) {
    if (wait.L <= 0 || history_size != wait.L || (int)prompt.size() < wait.lcp || wait.client != client || wait.sid != sid ||
        kvflash_prefix_hash(prompt, wait.lcp) != wait.prefix_hash) return pick;
    return {slot, std::max(pick.slot == slot ? pick.lcp : 0, wait.lcp)};
}

// ---- the slot's bundle bookkeeping -----------------------------------------------------------

// What the engine keeps per slot about the session's directory. A slot reused by another session
// starts it over (for_session): its head, generations and debts are never the previous session's.
struct KvFlashSlotBundle {
    KvFlashSlotHead head;            // the last park the writer acknowledged: what the directory holds (gen 0: none)
    KvFlashSlotHead staged;          // the park handed to the writer, the next save learns its verdict
    bool pending = false;
    bool rewrite = false;            // a park was skipped (its slab snapshot failed): the next one rewrites the slot whole
    std::vector<std::string> debt;   // files failed parks left (kvflash_bundle_leftovers), carried until a park commits
    uint32_t next_gen = 1;           // one counter for park generations and slab file ids: never reused within a session's directory
    int resume_L = 0;                // where the live turn resumed (0: cold): the park's dirty range starts there
    std::string key, sid, client;    // the session in this slot: its directory under sessions/, its identity
    int64_t created = 0;             // unix seconds of its first admit
    bool identity_durable = false;   // its meta.json (who it is) and the directory that names it are on disk: only then may a head follow

    static KvFlashSlotBundle for_session(std::string key, std::string sid, std::string client, int64_t created) {
        KvFlashSlotBundle b;
        b.key = std::move(key); b.sid = std::move(sid); b.client = std::move(client); b.created = created;
        return b;
    }
    // A session restored from its directory: the head it holds is what the next park builds on, and
    // the counter starts past every file id that head names.
    static KvFlashSlotBundle adopted(std::string key, std::string sid, std::string client, int64_t created, KvFlashSlotHead head) {
        KvFlashSlotBundle b = for_session(std::move(key), std::move(sid), std::move(client), created);
        b.identity_durable = true;   // it was restored from its directory
        b.next_gen = std::max({head.gen, head.end_gen, head.ckpt_gen, head.trunk_gen}) + 1;
        b.head = std::move(head);
        return b;
    }
    // The park writes a fresh directory state (new log) rather than appending to the head's log.
    bool fresh_park(bool failed) const { return head.gen == 0 || failed || rewrite; }
};

// Is the session's latest park on disk? `pending`: a park was staged and its verdict `failed` is
// in; `rewrite`: the last park was skipped. A session that is not durable is not evicted to disk:
// its state is newer than anything the directory holds.
inline bool kvflash_session_park_durable(bool pending, bool failed, bool rewrite, uint32_t head_gen) {
    return pending ? !failed : !rewrite && head_gen != 0;
}

// ---- meta.json and index.json -----------------------------------------------------------------

struct KvFlashSessionEntry {
    std::string key;                     // the directory name under sessions/
    std::string sid, client;
    std::string state = "disk";          // parked-gpu | disk | compact | exported | foreign (another server's or unreadable: listed, never touched)
    std::string archive_of, archive;     // P2 links (keys); written and read, not used yet
    std::string preamble;                // P1b: the preamble entry the session is bound to (hex16 of H_k), "" none
    int preamble_chunks = 0;             // and its k
    int64_t created = 0, last_used = 0;  // unix seconds
    uint64_t tokens = 0, bytes_live = 0, bytes_dead = 0, identity = 0;
    uint32_t gen = 0;                    // the head generation the fields describe (P3 claims; a meta behind the head is tolerated)
    std::vector<int32_t> history;        // in memory for sessions on disk only (the LCP pick), never written
    bool owner_pending = false;          // in memory only: the owner above was never made durable (a release kept the bundle); the next restore must publish it before any head
};

namespace detail {
inline void json_str(std::string & out, std::string_view s) {
    out += '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else out += (char)c;
    }
    out += '"';
}
inline void json_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xc0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3f)); }
    else { out += (char)(0xe0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3f)); out += (char)(0x80 | (cp & 0x3f)); }
}
// One flat JSON object: string, number and literal values, all returned as text.
inline bool json_flat_object(std::string_view s, std::map<std::string, std::string> & out) {
    size_t i = 0;
    auto ws = [&] { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; };
    auto str = [&](std::string & dst) {
        if (i >= s.size() || s[i] != '"') return false;
        for (++i; i < s.size(); ++i) {
            const char c = s[i];
            if (c == '"') { ++i; return true; }
            if (c != '\\') { dst += c; continue; }
            if (++i >= s.size()) return false;
            switch (s[i]) {
                case 'n': dst += '\n'; break;  case 't': dst += '\t'; break;  case 'r': dst += '\r'; break;
                case 'b': dst += '\b'; break;  case 'f': dst += '\f'; break;
                case 'u': {
                    if (i + 4 >= s.size()) return false;
                    const std::string hex(s.substr(i + 1, 4));
                    char * end = nullptr;
                    const unsigned long cp = std::strtoul(hex.c_str(), &end, 16);
                    if (end != hex.c_str() + 4) return false;
                    json_utf8(dst, (uint32_t)cp);
                    i += 4;
                    break;
                }
                default: dst += s[i];
            }
        }
        return false;
    };
    ws();
    if (i >= s.size() || s[i] != '{') return false;
    ++i; ws();
    auto end = [&] { ++i; ws(); return i == s.size(); };   // the object ends the text: nothing may follow its closing brace
    if (i < s.size() && s[i] == '}') return end();
    for (;;) {
        std::string key, val;
        ws();
        if (!str(key)) return false;
        ws();
        if (i >= s.size() || s[i] != ':') return false;
        ++i; ws();
        if (i < s.size() && s[i] == '"') { if (!str(val)) return false; }
        else { while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ' ' && s[i] != '\n') val += s[i++]; if (val.empty()) return false; }
        out[key] = val;
        ws();
        if (i >= s.size()) return false;
        if (s[i] == ',') { ++i; continue; }
        return s[i] == '}' && end();
    }
}
inline void json_fields(std::string & out, const KvFlashSessionEntry & e) {
    auto s = [&](const char * k, const std::string & v) { out += '"'; out += k; out += "\":"; json_str(out, v); out += ','; };
    auto n = [&](const char * k, unsigned long long v) { out += '"'; out += k; out += "\":" + std::to_string(v) + ','; };
    s("sid", e.sid); s("client", e.client); s("state", e.state); s("archive_of", e.archive_of); s("archive", e.archive);
    n("created", (unsigned long long)e.created); n("last_used", (unsigned long long)e.last_used); n("tokens", e.tokens);
    n("bytes_live", e.bytes_live); n("bytes_dead", e.bytes_dead); n("gen", e.gen);
    s("identity", kvflash_hex16(e.identity));
    if (!e.preamble.empty()) { s("preamble", e.preamble); n("preamble_chunks", (unsigned long long)e.preamble_chunks); }   // only a bound session: a store-off meta is what it was
    n("format", 1);
    out.back() = '}';
}
}  // namespace detail

inline std::string kvflash_meta_pack(const KvFlashSessionEntry & e) {
    std::string out = "{";
    detail::json_fields(out, e);
    return out;
}

inline bool kvflash_meta_unpack(std::string_view json, KvFlashSessionEntry & out) {
    std::map<std::string, std::string> m;
    if (!detail::json_flat_object(json, m) || m["sid"].empty()) return false;
    KvFlashSessionEntry e;
    auto num = [&](const char * k) { return std::strtoull(m[k].c_str(), nullptr, 10); };
    e.sid = m["sid"]; e.client = m["client"]; e.state = m["state"].empty() ? "disk" : m["state"];
    e.archive_of = m["archive_of"]; e.archive = m["archive"];
    e.created = (int64_t)num("created"); e.last_used = (int64_t)num("last_used"); e.tokens = num("tokens");
    e.bytes_live = num("bytes_live"); e.bytes_dead = num("bytes_dead"); e.gen = (uint32_t)num("gen");
    e.identity = std::strtoull(m["identity"].c_str(), nullptr, 16);
    e.preamble = m["preamble"]; e.preamble_chunks = (int)num("preamble_chunks");
    out = std::move(e);
    return true;
}

// ---- the compact tier (P3, SESSIONS_DESIGN 5.1) ----------------------------------------------------

constexpr const char * kCompactFile = "compact.json";

// Bytes of every regular file under `path` by stat (st_size, links not followed): what the store occupies.
inline uint64_t kvflash_tree_bytes(const std::string & path) {
    uint64_t n = 0;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it(path, ec), end; !ec && it != end; it.increment(ec)) {
        struct stat st{};
        if (it->is_regular_file(ec) && ::lstat(it->path().c_str(), &st) == 0) n += (uint64_t)st.st_size;
    }
    return n;
}

// The same count, but false when the walk hit an error (a directory vanished under it, no permission, a file that cannot be
// statted): the sum is then a partial one, so it is retried a few times and never reported as a size. `lstat_fn` is the seam
// tests fail.
using KvFlashLstat = std::function<int(const char *, struct stat *)>;
inline bool kvflash_tree_bytes_checked(const std::string & path, uint64_t & n, const KvFlashLstat & lstat_fn = [](const char * p, struct stat * st) { return ::lstat(p, st); }) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        uint64_t sum = 0;
        std::error_code ec;
        std::filesystem::recursive_directory_iterator it(path, ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            struct stat st{};
            const bool reg = std::filesystem::is_regular_file(it->symlink_status(ec));   // a link is not followed: a dangling one is no file and no error
            if (ec) break;
            if (reg) { if (lstat_fn(it->path().c_str(), &st) != 0) { ec = std::make_error_code(std::errc::io_error); break; } sum += (uint64_t)st.st_size; }
        }
        if (!ec) { n = sum; return true; }
    }
    return false;
}

struct KvFlashCompactDescriptor {
    uint32_t gen = 0;
    uint64_t tokens = 0;
    std::string history_sha;
    std::vector<std::string> keep, unlink;
};

inline std::string kvflash_compact_pack(const KvFlashCompactDescriptor & d) {
    auto arr = [](const std::vector<std::string> & v) {
        std::string o = "[";
        for (const auto & x : v) { if (o.size() > 1) o += ','; detail::json_str(o, x); }
        return o + "]";
    };
    return "{\"gen\":" + std::to_string(d.gen) + ",\"tokens\":" + std::to_string(d.tokens) + ",\"history_sha\":\"" + d.history_sha +
           "\",\"keep\":" + arr(d.keep) + ",\"unlink\":" + arr(d.unlink) + "}";
}

// The whole text must be one JSON object of exactly the descriptor's fields: gen in 32 bits, tokens in 64, the hash and the
// two name lists strings (strict parse: a tail, a wrong type or an out-of-range number refuses it).
inline bool kvflash_compact_unpack(std::string_view json, KvFlashCompactDescriptor & out) {
    const nlohmann::json j = nlohmann::json::parse(json.begin(), json.end(), nullptr, false);
    if (j.is_discarded() || !j.is_object()) return false;
    KvFlashCompactDescriptor d;
    const auto g = j.find("gen"), t = j.find("tokens"), h = j.find("history_sha"), k = j.find("keep"), u = j.find("unlink");
    if (g == j.end() || t == j.end() || h == j.end() || k == j.end() || u == j.end()) return false;
    if (!g->is_number_unsigned() || g->get<uint64_t>() > std::numeric_limits<uint32_t>::max() || !t->is_number_unsigned() || !h->is_string() || h->get<std::string>().empty()) return false;
    for (const auto * arr : {&*k, &*u}) {
        if (!arr->is_array()) return false;
        for (const auto & x : *arr) if (!x.is_string()) return false;
    }
    d.gen = g->get<uint32_t>(); d.tokens = t->get<uint64_t>(); d.history_sha = h->get<std::string>();
    d.keep = k->get<std::vector<std::string>>(); d.unlink = u->get<std::vector<std::string>>();
    out = std::move(d);
    return true;
}

// The descriptor's check of a history: the fnv of its token bytes, hex.
inline std::string kvflash_history_sha(const std::vector<int32_t> & history) {
    return kvflash_hex16(kvflash_fnv1a(std::string_view((const char *)history.data(), history.size() * 4)));
}

// A name that stays inside its session directory: one path component, no control bytes.
inline bool kvflash_safe_file_name(const std::string & n) {
    if (n.empty() || n.size() > 255 || n == "." || n == "..") return false;
    return std::none_of(n.begin(), n.end(), [](unsigned char c) { return c == '/' || c == '\\' || c < 0x20; });
}

// The descriptor in `dir` is trusted only when it parses and describes this head: its generation, its token
// count and the hash of its history, with every file it lists a plain name that is not the head, the meta or
// the descriptor itself. False for a missing, empty, malformed, stale or foreign descriptor.
inline bool kvflash_compact_validate(const std::string & dir, const KvFlashSlotHead & head, KvFlashCompactDescriptor & out) {
    std::vector<uint8_t> b;
    KvFlashCompactDescriptor d;
    if (!kvflash_bundle_read(dir + "/" + kCompactFile, b, 1 << 20) || !kvflash_compact_unpack(std::string_view((const char *)b.data(), b.size()), d)) return false;
    if (d.gen != head.gen || d.tokens != head.history.size() || d.history_sha != kvflash_history_sha(head.history)) return false;
    if (std::find(d.keep.begin(), d.keep.end(), "head") == d.keep.end()) return false;
    for (const std::string & f : d.unlink)
        if (!kvflash_safe_file_name(f) || f == "head" || f == "meta.json" || f == kCompactFile) return false;
    out = std::move(d);
    return true;
}

// Every file in `dir` the compact tier does not keep: the log, the slabs, a leftover of a park or a tmp.
inline std::vector<std::string> kvflash_compact_leftovers(const std::string & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto & de : std::filesystem::directory_iterator(dir, ec)) {
        const std::string n = de.path().filename().string();
        if (n != "head" && n != "meta.json" && n != "exported" && n != kCompactFile) out.push_back(n);
    }
    std::sort(out.begin(), out.end());
    return out;
}

struct KvFlashCompactResult {
    bool committed = false;               // compact.json is durable: the tier is decided
    bool crashed = false;                 // a crash point fired (tests)
    size_t unlinks = 0;                   // files the descriptor lists
    std::vector<std::string> left;        // listed files whose unlink failed: the steward retries them
};

// The transition: compact.json (tokens, rungs, what goes) is committed and its directory fsynced
// BEFORE any unlink, then the files go; the caller rewrites meta.state after. head and meta.json stay.
// A crash anywhere leaves a directory the scan reads as compact (descriptor) or as the old disk
// session (no descriptor), never half of each. crash_at (tests): 1 before the descriptor, 2 after
// it, 2+i after the i-th unlink.
template <class Unlink>
inline KvFlashCompactResult kvflash_compact_dir(const std::string & dir, const KvFlashSlotHead & head, Unlink && unlink_file, int crash_at = 0) {
    KvFlashCompactResult r;
    KvFlashCompactDescriptor d;
    d.gen = head.gen; d.tokens = head.history.size();
    d.history_sha = kvflash_history_sha(head.history);
    d.keep = {"head", "meta.json"};
    d.unlink = kvflash_compact_leftovers(dir);
    r.unlinks = d.unlink.size();
    if (crash_at == 1) { r.crashed = true; return r; }
    const std::string json = kvflash_compact_pack(d);
    if (!kvflash_bundle_write(dir + "/" + kCompactFile, std::vector<uint8_t>(json.begin(), json.end())) || !kvflash_fsync_dir(dir)) return r;
    r.committed = true;
    if (crash_at == 2) { r.crashed = true; return r; }
    for (size_t i = 0; i < d.unlink.size(); ++i) {
        if (!unlink_file(dir + "/" + d.unlink[i])) r.left.push_back(d.unlink[i]);
        if (crash_at == (int)(3 + i)) { r.crashed = true; return r; }
    }
    kvflash_fsync_dir(dir);
    return r;
}

// ---- scan and migration -----------------------------------------------------------------------

struct KvFlashScanReport {
    std::vector<std::string> removed;   // a directory with nothing committed in it
    std::vector<std::string> skipped;   // a head of another server or a bad format: left as it is, not listed
    std::vector<std::string> failed;    // a metadata write or fsync that did not land (keys): retried by the next start
};

namespace detail {
inline bool is_session_key(const std::string & name) { return !name.empty() && name.find('.') == std::string::npos; }
inline bool is_quarantine_name(const std::string & n) { return n.find(".quarantine.") != std::string::npos; }
inline bool is_slot_dir_name(const std::string & n) { return n.size() > 4 && n.compare(0, 4, "slot") == 0 && n.find_first_not_of("0123456789", 4) == std::string::npos; }
inline std::string migrated_sid(const std::vector<int32_t> & history) {
    return "migrated:" + kvflash_hex16(kvflash_fnv1a(std::string_view((const char *)history.data(), history.size() * 4)));
}
// exists() with the I/O fault seam; an error is reported in ec and is never the same as "absent".
inline bool path_exists(const std::string & path, std::error_code & ec) {
    ec.clear();
    if (kvflash_io_fails("stat", path)) { ec = std::make_error_code(std::errc::io_error); return false; }
    return std::filesystem::exists(path, ec);
}
// meta.json of a directory, durably: the file, then the directory that names it.
inline bool write_meta_file(const std::string & dir, const KvFlashSessionEntry & e) {
    const std::string json = kvflash_meta_pack(e);
    return json.size() <= kKvFlashMetaMax && kvflash_bundle_write(dir + "/meta.json", std::vector<uint8_t>(json.begin(), json.end())) && kvflash_fsync_dir(dir);
}
inline int64_t file_mtime(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 ? (int64_t)st.st_mtime : 0;
}
inline uint64_t file_size_or0(const std::string & path) {
    std::error_code ec;
    const auto n = std::filesystem::file_size(path, ec);
    return ec ? 0 : (uint64_t)n;
}
// The identity a head file names, whatever it is: a head of another server is still someone's.
inline uint64_t peek_head_identity(const std::string & dir) {
    std::vector<uint8_t> b;
    uint64_t id = 0;
    if (kvflash_bundle_read(dir + "/head", b, 1 << 30) && b.size() >= 16 && std::memcmp(b.data(), kHeadMagic, 8) == 0) std::memcpy(&id, b.data() + 8, 8);
    return id;
}
inline uint64_t dir_bytes(const std::string & dir) {
    uint64_t n = 0;
    std::error_code ec;
    for (const auto & de : std::filesystem::directory_iterator(dir, ec)) if (de.is_regular_file(ec)) n += de.file_size(ec);
    return n;
}
inline KvFlashSessionEntry foreign_entry(const std::string & key, const std::string & dir) {
    KvFlashSessionEntry e;
    std::vector<uint8_t> meta;
    if (!kvflash_bundle_read(dir + "/meta.json", meta, kKvFlashMetaMax) || !kvflash_meta_unpack(std::string_view((const char *)meta.data(), meta.size()), e)) {
        e = KvFlashSessionEntry{};
        e.sid = "foreign:" + key; e.client = "foreign"; e.created = e.last_used = file_mtime(dir + "/head");
    }
    e.key = key; e.state = "foreign"; e.identity = peek_head_identity(dir); e.bytes_live = dir_bytes(dir); e.bytes_dead = 0; e.tokens = 0;
    return e;
}
inline bool read_head(const std::string & dir, uint64_t identity, size_t head_max, KvFlashSlotHead & h) {
    std::vector<uint8_t> bytes;
    return kvflash_bundle_read(dir + "/head", bytes, head_max) && kvflash_head_unpack(bytes.data(), bytes.size(), identity, h);
}
}  // namespace detail

inline std::string kvflash_sessions_dir(const std::string & root) { return root + "/sessions"; }

// Every session under <root>/sessions/, from its directory: a loadable head (the truth: tokens,
// history, sizes) and meta.json (who it is). A directory with no head holds nothing committed and is
// removed. Everything else that cannot be proven this server's and restorable is listed `foreign` and
// left as it is: a head of another server, a head or a meta.json that cannot be read (a named record
// never becomes an ownerless one), a failed filesystem query (unknown is not absent), and a log or slab
// that fails its checks (streamed here, so no record is eligible for a policy before it was read once).
// The rest are `disk` (nothing is in a slot at start), newest first. Metadata repairs that did not land
// are named in rep.failed.
inline std::vector<KvFlashSessionEntry> kvflash_session_scan(const std::string & root, uint64_t identity, size_t record_bytes,
                                                              size_t head_max, KvFlashScanReport & rep, size_t slab_max = SIZE_MAX) {
    namespace fs = std::filesystem;
    std::vector<KvFlashSessionEntry> out;
    std::error_code ec;
    std::vector<std::string> names;
    for (const auto & de : fs::directory_iterator(kvflash_sessions_dir(root), ec))
        if (de.is_directory(ec) && detail::is_session_key(de.path().filename().string())) names.push_back(de.path().filename().string());
    std::sort(names.begin(), names.end());
    // what the migration set aside: listed as foreign, never read, never removed
    for (const auto & de : fs::directory_iterator(kvflash_sessions_dir(root), ec))
        if (de.is_directory(ec) && detail::is_quarantine_name(de.path().filename().string())) {
            rep.skipped.push_back(de.path().filename().string());
            out.push_back(detail::foreign_entry(de.path().filename().string(), de.path().string()));
        }
    const size_t chunk_bytes = record_bytes > kKvFlashRecordTag ? record_bytes - kKvFlashRecordTag : 0;
    auto set_aside = [&](const std::string & key, const std::string & dir) { rep.skipped.push_back(key); out.push_back(detail::foreign_entry(key, dir)); };
    for (const std::string & key : names) {
        const std::string dir = kvflash_sessions_dir(root) + "/" + key;
        const bool has_head = detail::path_exists(dir + "/head", ec);
        if (ec) { set_aside(key, dir); continue; }   // the query failed: unknown, not "nothing committed"
        if (!has_head) { fs::remove_all(dir, ec); rep.removed.push_back(key); continue; }
        KvFlashSlotHead h;
        if (!detail::read_head(dir, identity, head_max, h)) { set_aside(key, dir); continue; }
        // a descriptor makes a directory compact only when it describes this head; any other marker leaves the
        // record foreign with its reconstruction files untouched
        const bool marker = fs::symlink_status(dir + "/" + kCompactFile, ec).type() != fs::file_type::not_found;
        KvFlashCompactDescriptor desc;
        const bool compact = marker && kvflash_compact_validate(dir, h, desc);
        if (marker && !compact) { set_aside(key, dir); continue; }
        KvFlashSessionEntry e;
        std::vector<uint8_t> meta;
        if (!kvflash_bundle_read(dir + "/meta.json", meta, kKvFlashMetaMax) ||
            !kvflash_meta_unpack(std::string_view((const char *)meta.data(), meta.size()), e)) { set_aside(key, dir); continue; }
        if (compact) {   // never a bundle load: the history is the head's, what the descriptor lists is finished off idempotently
            for (const std::string & f : desc.unlink) if (f != "exported") fs::remove_all(dir + "/" + f, ec);   // the validated names only; an older build's descriptor may list the export marker
            kvflash_fsync_dir(dir);
            const bool was = e.state == "compact";
            e.key = key; e.state = "compact"; e.identity = identity; e.tokens = h.history.size(); e.gen = h.gen;
            e.bytes_live = kvflash_tree_bytes(dir); e.bytes_dead = 0;
            if (!was) { const std::string json = kvflash_meta_pack(e); kvflash_bundle_write(dir + "/meta.json", std::vector<uint8_t>(json.begin(), json.end())); }
            out.push_back(std::move(e));
            continue;
        }
        std::string why;
        if (!kvflash_bundle_validate(dir, h, chunk_bytes, slab_max, why)) {
            std::fprintf(stderr, "[kvflash] session %s: %s (listed as foreign, left in place)\n", key.c_str(), why.c_str());
            set_aside(key, dir);
            continue;
        }
        size_t live = 0;
        for (uint64_t off : h.map) live += off != 0;
        const uint64_t records = h.log_bytes > kKvFlashLogHeader ? h.log_bytes - kKvFlashLogHeader : 0;
        e.key = key; e.state = "disk"; e.identity = identity; e.tokens = h.history.size(); e.gen = h.gen;
        e.bytes_live = (uint64_t)live * record_bytes + detail::file_size_or0(dir + "/head");
        for (uint32_t id : {h.end_gen, h.ckpt_gen, h.trunk_gen})
            if (id) e.bytes_live += detail::file_size_or0(dir + "/" + kvflash_bundle_slab_name(id));
        e.bytes_dead = records > (uint64_t)live * record_bytes ? records - (uint64_t)live * record_bytes : 0;
        e.history = std::move(h.history);
        out.push_back(std::move(e));
    }
    // what the migration left in place: slot<i>/ directories of another server, or unreadable
    for (const auto & de : fs::directory_iterator(root, ec)) {
        const std::string n = de.path().filename().string();
        if (detail::is_slot_dir_name(n) && de.is_directory(ec)) {
            const bool has_head = detail::path_exists((de.path() / "head").string(), ec);
            if (has_head || ec) set_aside(n, de.path().string());
        }
    }
    // a successor's archive_of is the truth of the link: an archive whose publication a crash cut off is repaired
    auto repair = [&](const KvFlashSessionEntry & old) {
        if (!detail::write_meta_file(kvflash_sessions_dir(root) + "/" + old.key, old)) rep.failed.push_back(old.key);
    };
    for (const auto & succ : out)
        if (!succ.archive_of.empty())
            for (auto & old : out)
                if (old.key == succ.archive_of && old.archive.empty()) {
                    old.archive = succ.key;
                    if (old.state != "foreign") repair(old);
                }
    // an archive naming a successor that never committed (its directory is gone) points at nothing
    for (auto & old : out)
        if (!old.archive.empty() && old.state != "foreign" &&
            std::none_of(out.begin(), out.end(), [&](const auto & e) { return e.key == old.archive; })) {
            old.archive.clear();
            repair(old);
        }
    std::stable_sort(out.begin(), out.end(), [](const auto & a, const auto & b) { return a.last_used > b.last_used; });
    return out;
}

// Today's <root>/slot<i>/ become sessions/<key>/ with sid "migrated:<fnv(history)>". The meta.json
// (stamped with the head's mtime: the slots' recency survives) is written and fsynced INSIDE slot<i>/
// first and the directory renamed after, so a directory under sessions/ with a head and no meta.json
// was never a proven migration. A slot directory with no head holds an interrupted first park and is
// cleaned. One whose head this server cannot load (another model or KV type, a smaller context,
// damage, or a filesystem query that failed) is never erased: it is renamed to
// sessions/<name>.quarantine.<n> (the first free n, an earlier quarantine is never replaced), which
// the scan lists as foreign; only an explicit request deletes it. A rename is atomic inside one
// filesystem, and a directory already moved is gone from the list, so a restart finishes whatever
// a crash cut off and moves nothing twice. A meta that did not land durably leaves the slot
// directory where it is (named in rep.failed, retried by the next start). `crash_after` (tests): return
// after that many steps (the sessions/ mkdir, then each meta write and each rename count one).
// Returns the number migrated.
inline size_t kvflash_migrate_slot_dirs(const std::string & root, uint64_t identity, size_t head_max, KvFlashScanReport & rep, int crash_after = 0) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<std::string> slots;
    for (const auto & de : fs::directory_iterator(root, ec)) {
        const std::string n = de.path().filename().string();
        if (detail::is_slot_dir_name(n) && de.is_directory(ec)) slots.push_back(n);
    }
    if (slots.empty()) return 0;
    std::sort(slots.begin(), slots.end());
    const std::string sdir = kvflash_sessions_dir(root);
    int step = 0;
    auto crashed = [&] { return crash_after > 0 && ++step >= crash_after; };
    fs::create_directories(sdir, ec);
    if (crashed()) return 0;
    std::set<std::string> taken;
    for (const auto & de : fs::directory_iterator(sdir, ec)) taken.insert(de.path().filename().string());
    size_t moved = 0;
    for (const std::string & name : slots) {
        const std::string dir = root + "/" + name;
        const bool has_head = detail::path_exists(dir + "/head", ec);
        const bool unknown = (bool)ec;
        if (!has_head && !unknown) {   // nothing was committed here
            kvflash_bundle_clean(dir, nullptr);
            fs::remove(dir, ec);
            rep.removed.push_back(name);
            continue;
        }
        KvFlashSlotHead h;
        bool loadable = false;
        if (!unknown && detail::read_head(dir, identity, head_max, h)) {
            const bool has_log = detail::path_exists(dir + "/" + kvflash_bundle_log_name(h.log_gen), ec);
            loadable = has_log && !ec;
        }
        KvFlashSessionEntry e;
        if (loadable) {
            e.sid = detail::migrated_sid(h.history); e.client = "migrated"; e.state = "disk";
            e.created = e.last_used = detail::file_mtime(dir + "/head");
            e.tokens = h.history.size(); e.identity = identity;
            e.key = kvflash_session_dir_key(e.client, e.sid, taken);
            if (!detail::write_meta_file(dir, e)) { rep.failed.push_back(name); continue; }   // the proof of the migration is durable before the rename
            if (crashed()) return moved;
        } else {
            int n = 1;
            while (taken.count(name + ".quarantine." + std::to_string(n))) ++n;
            e.key = name + ".quarantine." + std::to_string(n);
        }
        fs::rename(dir, sdir + "/" + e.key, ec);
        if (ec) { rep.skipped.push_back(name); continue; }
        taken.insert(e.key);
        if (!loadable) rep.skipped.push_back(name); else ++moved;
        if (crashed()) return moved;
    }
    if (!kvflash_fsync_dir(sdir) || !kvflash_fsync_dir(root)) rep.failed.push_back("sessions");
    return moved;
}

// ---- the store --------------------------------------------------------------------------------

class KvFlashSessionStore {
public:
    explicit KvFlashSessionStore(std::string root) : root_(std::move(root)) {
        std::random_device rd;
        char b[17];
        std::snprintf(b, sizeof b, "%08x%08x", rd(), rd());
        boot_ = b;
    }

    const std::string & root() const { return root_; }
    std::string dir(const std::string & key) const { return kvflash_sessions_dir(root_) + "/" + key; }
    std::vector<KvFlashSessionEntry> & entries() { return entries_; }
    const std::vector<KvFlashSessionEntry> & entries() const { return entries_; }
    // Names this instance of the session under `key`, "" when there is none: it changes when the key is deleted and used again (even in
    // the same second, even by an import of the same archive) and in every process, so a delayed retry of a destructive call that carries
    // an old one is refused instead of landing on the replacement. In memory only: across a restart every token is new (fails safe).
    std::string incarnation(const std::string & key) const {
        if (!find_key(key)) return "";
        const auto it = erased_.find(key);
        return boot_ + "." + std::to_string(it == erased_.end() ? 0 : it->second);
    }

    const KvFlashSessionEntry * find_key(const std::string & key) const {
        for (const auto & e : entries_) if (e.key == key) return &e;
        return nullptr;
    }
    // The keys that are taken: every entry's, and every key held for a directory a worker publishes later (an import).
    std::set<std::string> keys() const {
        std::set<std::string> k = reserved_;
        for (const auto & e : entries_) k.insert(e.key);
        return k;
    }
    std::string new_key(const std::string & client, const std::string & sid) const { return kvflash_session_dir_key(client, sid, keys()); }
    // One reservation for every allocator (a cold admission, an import): the key is held, and new_key skips it, until released.
    // `also_taken`: names the caller knows are in use (directories on the disk).
    std::string reserve_key(const std::string & client, const std::string & sid, std::set<std::string> also_taken = {}) {
        for (const std::string & k : keys()) also_taken.insert(k);
        std::string key = kvflash_session_dir_key(client, sid, also_taken);
        reserved_.insert(key);
        return key;
    }
    void release_key(const std::string & key) { reserved_.erase(key); }

    // Adds an entry under a key nothing lists yet; false (and nothing changes) when a live entry holds it.
    bool insert_new(KvFlashSessionEntry e) {
        if (find_key(e.key)) return false;
        entries_.push_back(std::move(e));
        return true;
    }
    void upsert(KvFlashSessionEntry e) {
        for (auto & x : entries_) if (x.key == e.key) { x = std::move(e); return; }
        entries_.push_back(std::move(e));
    }

    void erase(const std::string & key) {
        if (find_key(key)) ++erased_[key];
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const auto & e) { return e.key == key; }), entries_.end());
    }

    // P7's hook (4.3), no recall yet: the earlier instances of a session reachable from `new_key` by archive_of, nearest
    // first. An archived instance is on disk, so `history` is its tokens; its chunk map is the head in `dir`.
    struct RecallCandidate { std::string old_key, dir; std::vector<int32_t> history; };
    std::vector<RecallCandidate> recall_candidates(const std::string & new_key) const {
        std::vector<RecallCandidate> out;
        std::set<std::string> seen{new_key};
        const KvFlashSessionEntry * e = find_key(new_key);
        while (e && !e->archive_of.empty() && seen.insert(e->archive_of).second) {
            e = find_key(e->archive_of);
            if (e && e->state != "foreign") out.push_back({e->key, dir(e->key), e->history});
        }
        return out;
    }

    // The current instance of a session: the one no successor has archived. A sid starting again
    // leaves the old instance in its directory and a successor in its own.
    const KvFlashSessionEntry * current(const std::string & client, const std::string & sid) const {
        const KvFlashSessionEntry * best = nullptr;
        for (const auto & e : entries_)
            if (e.client == client && e.sid == sid && e.state != "foreign" && e.archive.empty() && (!best || e.last_used >= best->last_used)) best = &e;
        return best;
    }
    // Publish the link between an instance and its successor: the successor's meta (archive_of) is
    // durable first, then the old one's (archive). A crash between is repaired by the scan.
    bool link_archive(const std::string & old_key, const std::string & new_key) {
        KvFlashSessionEntry * o = nullptr;
        for (auto & e : entries_) if (e.key == old_key) o = &e;
        const KvFlashSessionEntry * n = find_key(new_key);
        if (!o || !n || n->archive_of != old_key || o->archive == new_key) return false;
        o->archive = new_key;
        return write_meta(old_key);
    }

    bool write_meta(const std::string & key) const {
        const KvFlashSessionEntry * e = find_key(key);
        return e && detail::write_meta_file(dir(key), *e);   // false too for a meta a reader would refuse (over kKvFlashMetaMax)
    }
    // index.json: a cache of every meta.json (plus the key), for readers outside the engine.
    std::string index_json() const {
        std::string out = "[";
        for (const auto & e : entries_) {
            if (out.size() > 1) out += ',';
            out += "{\"key\":";
            detail::json_str(out, e.key);
            out += ',';
            detail::json_fields(out, e);
        }
        out += "]";
        return out;
    }
    bool write_index() const {
        const std::string out = index_json();
        return kvflash_bundle_write(root_ + "/index.json", std::vector<uint8_t>(out.begin(), out.end())) && kvflash_fsync_dir(root_);
    }

    // The session leaves its slot; its bundle stays where it is and its tokens move to memory.
    // Refused without a committed head in the directory: nothing is listed as on disk that is not.
    // `acknowledged`: the writer acknowledged that head, so a failed existence query does not decide (the entry still becomes disk).
    bool evict(const std::string & key, std::vector<int32_t> history, bool acknowledged = false) {
        std::error_code ec;
        if (!acknowledged && !std::filesystem::exists(dir(key) + "/head", ec)) return false;
        for (auto & e : entries_) if (e.key == key) {
            e.state = "disk"; e.tokens = history.size(); e.history = std::move(history);
            return write_meta(key) && write_index();
        }
        return false;
    }
    void mark_owner_pending(const std::string & key, bool pending) {
        for (auto & e : entries_) if (e.key == key) { e.owner_pending = pending; return; }
    }
    // This server cannot restore the session (its bundle is refused): listed, never touched.
    bool mark_foreign(const std::string & key) {
        for (auto & e : entries_) if (e.key == key) { e.state = "foreign"; std::vector<int32_t>().swap(e.history); return write_index(); }
        return false;
    }
    // The session came back into a slot: its history lives in the engine again.
    bool adopt(const std::string & key) {
        for (auto & e : entries_) if (e.key == key) {
            e.state = "parked-gpu"; std::vector<int32_t>().swap(e.history);
            return write_meta(key) && write_index();
        }
        return false;
    }

    // The disk session a request restores. An explicit request: the newest instance of its own
    // (client, sid) that shares a token with the prompt, else the ownerless (migrated or anon)
    // session sharing the longest prefix beyond the preamble (lcp > e_role). An anonymous request:
    // the latter only. Content never overrides a stated identity.
    const KvFlashSessionEntry * pick_disk(const std::string & client, const std::string & sid, const std::vector<int32_t> & prompt, int e_role) const {
        auto lcp = [&](const KvFlashSessionEntry & e) {
            const size_t cap = std::min(prompt.size(), e.history.size());
            size_t l = 0;
            while (l < cap && prompt[l] == e.history[l]) ++l;
            return l;
        };
        const KvFlashSessionEntry * best = nullptr;
        if (kvflash_client_explicit(client))
            for (const auto & e : entries_)
                if (e.state == "disk" && e.client == client && e.sid == sid && lcp(e) > 0 && (!best || e.last_used >= best->last_used)) best = &e;
        if (best) return best;
        size_t best_lcp = 0;
        for (const auto & e : entries_) {
            if (e.state != "disk" || kvflash_client_explicit(e.client) || e.client == "foreign") continue;
            const size_t l = lcp(e);
            if (l <= (size_t)std::max(e_role, 0)) continue;
            if (l > best_lcp || (l == best_lcp && e.last_used > best->last_used)) { best = &e; best_lcp = l; }
        }
        return best;
    }

private:
    std::string root_;
    std::vector<KvFlashSessionEntry> entries_;
    std::string boot_;                          // per process (incarnation())
    std::map<std::string, uint32_t> erased_;    // how often each key was deleted in this process (incarnation())
    std::set<std::string> reserved_;   // keys held for a directory not yet listed (owner thread only)
};

// ---- the slot table: who owns each slot, and the transitions the engine makes on it ------------

enum class KvFlashRestorePick { none, busy, load };          // what the request needs before it is admitted
struct KvFlashRestorePlan { KvFlashRestorePick pick = KvFlashRestorePick::none; std::string key; };
enum class KvFlashRestore { absent, restored, busy, refused };   // what restoring a disk session came to

// What an adopt came to, as the restore reports it: a ledger that lacks room stays Busy (the request waits), it never becomes a cold admit.
inline KvFlashRestore kvflash_restore_of(KvFlashAdopt a) {
    return a == KvFlashAdopt::Ok ? KvFlashRestore::restored : a == KvFlashAdopt::Busy ? KvFlashRestore::busy : KvFlashRestore::refused;
}

struct KvFlashParkStats { size_t tokens = 0; uint64_t bytes_live = 0, bytes_dead = 0; std::string preamble; int preamble_chunks = 0; };

// The engine's per-slot session bookkeeping and the session store behind it, with no GPU in it: every
// transition the engine makes on ownership (a cold admit, an adoption, an eviction, a park's metadata)
// is a method here, so the tests drive the same code the engine runs.
class KvFlashSessionBook {
public:
    void init(int slots) { slot_.assign((size_t)slots, KvFlashSlotBundle{}); }
    // Persistence on: sessions live under store.root()/sessions/. Without it ownership is still kept in memory.
    void enable_store(KvFlashSessionStore store, uint64_t identity) { store_ = std::move(store); identity_ = identity; persistent_ = true; }
    bool persistent() const { return persistent_; }
    uint64_t identity() const { return identity_; }
    KvFlashSessionStore & store() { return store_; }
    const KvFlashSessionStore & store() const { return store_; }
    KvFlashSlotBundle & operator[](int slot) { return slot_[(size_t)slot]; }
    const KvFlashSlotBundle & operator[](int slot) const { return slot_[(size_t)slot]; }
    int slots() const { return (int)slot_.size(); }

    // The archive_of an admit names for the instance it evicted: its key once it is a disk instance with a committed head,
    // "" (no link) when the eviction failed or dropped it. A link to an instance with no durable head is never published.
    std::string archive_target(const std::string & key) const {
        const KvFlashSessionEntry * e = store_.find_key(key);
        return e && e->state == "disk" ? key : std::string();
    }
    // P7's hook through the book: an instance a startup adoption put back into a slot gave its history to the slot's head.
    std::vector<KvFlashSessionStore::RecallCandidate> recall_candidates(const std::string & new_key) const {
        auto out = store_.recall_candidates(new_key);
        for (auto & c : out)
            if (c.history.empty())
                for (const KvFlashSlotBundle & sb : slot_) if (sb.key == c.old_key) c.history = sb.head.history;
        return out;
    }

    // A cold admit: the slot starts the session over. Its owner is kept in memory whatever the
    // persistence; the directory, the store entry and the archive link only when it is on. The instance it
    // archives is the sid's current one unless the admit names it: a key (a compaction), or "" for none (a
    // summary-generation request is no successor of the instance it ran beside).
    void begin_session(int slot, const std::string & client, const std::string & sid, int64_t now, std::optional<std::string> archive_of = std::nullopt) {
        KvFlashSlotBundle nb;
        nb.sid = sid; nb.client = client; nb.created = now;
        if (!persistent_) { slot_[(size_t)slot] = std::move(nb); return; }
        nb.key = store_.new_key(nb.client, nb.sid);
        KvFlashSessionEntry e;
        e.key = nb.key; e.sid = nb.sid; e.client = nb.client; e.state = "parked-gpu"; e.created = e.last_used = nb.created; e.identity = identity_;
        const KvFlashSessionEntry * prev = store_.current(nb.client, nb.sid);   // the sid starts again: its earlier instance stays in its own directory
        if (archive_of) e.archive_of = *archive_of;
        else if (prev) e.archive_of = prev->key;
        store_.upsert(std::move(e));
        slot_[(size_t)slot] = std::move(nb);
    }
    // The slot holds no session any more.
    void reset(int slot) { slot_[(size_t)slot] = KvFlashSlotBundle{}; }

    // A session loaded from sessions/<key>/ into the slot: its bookkeeping is the head it holds, history
    // included (an eviction before the next park lists the session with it). Returns the history the
    // parked session starts with.
    std::vector<int32_t> adopt(int slot, const std::string & key, KvFlashSlotHead head) {
        std::vector<int32_t> history = head.history;
        const KvFlashSessionEntry * e = store_.find_key(key);
        slot_[(size_t)slot] = KvFlashSlotBundle::adopted(key, e ? e->sid : "", e ? e->client : "", e ? e->created : 0, std::move(head));
        if (e && e->owner_pending) slot_[(size_t)slot].identity_durable = false;   // a released session whose owner never landed
        store_.adopt(key);
        return history;
    }

    // An explicit request resumes an ownerless session (a migrated or anonymous one): it becomes that client's, the directory (named
    // after the old owner) stays. A new publication of identity: a head may follow only once the new owner's meta.json is durable.
    // False when nothing changed (the owner is explicit already, or the request is not).
    bool promote_owner(int slot, const std::string & client, const std::string & sid) {
        KvFlashSlotBundle & sb = slot_[(size_t)slot];
        if (kvflash_client_explicit(sb.client) || !kvflash_client_explicit(client)) return false;
        sb.client = client; sb.sid = sid;
        sb.identity_durable = false;
        return true;
    }

    // The head an eviction leaves on disk: the staged one when the writer took it, else the last acknowledged.
    const KvFlashSlotHead * committed_head(int slot, bool failed) const {
        const KvFlashSlotBundle & sb = slot_[(size_t)slot];
        const bool staged_ok = sb.pending && !failed;
        return staged_ok ? &sb.staged : sb.head.gen != 0 ? &sb.head : nullptr;
    }
    // The session leaves the slot, its bundle stays: listed as disk with the history its head holds.
    bool evict(int slot, bool failed) {
        const KvFlashSlotHead * on_disk = durable(slot, failed) ? committed_head(slot, failed) : nullptr;   // a head older than the session is not the session
        if (!on_disk) return false;
        const bool ok = store_.evict(slot_[(size_t)slot].key, on_disk->history);
        slot_[(size_t)slot] = KvFlashSlotBundle{};
        return ok;
    }

    // A park the disk refused: the session leaves the slot and its host memory but the directory keeps what the writer last
    // acknowledged, listed as disk with that head's history (the next turn restores it or prefills cold). False only when there is no
    // acknowledged head: the caller drops the session. A store or index write that fails does not make the head any less durable: the
    // slot is still released and the directory stays; the entry is disk in memory and the next successful write or the startup scan
    // repairs the files.
    bool release_to_last_commit(int slot) {
        KvFlashSlotBundle & sb = slot_[(size_t)slot];
        if (sb.key.empty() || sb.head.gen == 0) return false;
        const bool written = store_.evict(sb.key, sb.head.history, /*acknowledged=*/true);
        if (!written)
            std::fprintf(stderr, "[kvflash] session %s: released with its committed bundle kept, the store or index write failed (repaired by the next write or scan)\n", sb.key.c_str());
        store_.mark_owner_pending(sb.key, !written && !sb.identity_durable);   // an owner that is on no disk stays unpublished across the restore
        slot_[(size_t)slot] = KvFlashSlotBundle{};
        return true;
    }

    // Before a park is handed to the writer: the session's metadata. The first park of a directory needs
    // its identity (meta.json and the directories naming it) durable, or the head would outlive an owner
    // nobody can prove; false then (the owner stays, the park is owed whole, nothing is submitted). Once
    // the identity is durable a failed refresh (last_used, sizes) does not stop a park: the head is the truth.
    bool publish_park(int slot, const KvFlashParkStats & st, int64_t now) {
        KvFlashSlotBundle & sb = slot_[(size_t)slot];
        KvFlashSessionEntry e;
        if (const KvFlashSessionEntry * old = store_.find_key(sb.key)) e = *old;
        e.key = sb.key; e.sid = sb.sid; e.client = sb.client; e.state = "parked-gpu"; e.identity = identity_;
        e.created = sb.created; e.last_used = now; e.tokens = st.tokens; e.bytes_live = st.bytes_live; e.bytes_dead = st.bytes_dead;
        e.preamble = st.preamble; e.preamble_chunks = st.preamble_chunks;   // the preamble binding (sessions P1b), "" = none
        store_.upsert(std::move(e));
        std::error_code mk;
        const bool created = std::filesystem::create_directory(store_.dir(sb.key), mk);
        bool ok = !mk;
        if (ok && (created || !sb.identity_durable)) ok = kvflash_fsync_dir(store_.dir(""));
        ok = ok && store_.write_meta(sb.key);
        if (ok) { sb.identity_durable = true; store_.mark_owner_pending(sb.key, false); }
        const KvFlashSessionEntry * cur = store_.find_key(sb.key);
        if (ok && cur && !cur->archive_of.empty()) store_.link_archive(cur->archive_of, sb.key);   // the successor's meta is durable: publish the old instance's link
        if (!ok || !store_.write_index()) std::fprintf(stderr, "[kvflash] session %s: meta or index not written\n", sb.key.c_str());
        if (!ok && !sb.identity_durable) { sb.rewrite = true; return false; }
        return true;
    }

    // Is (client, sid) running a request in a slot now? One live owner per explicit identity: a second
    // request for it waits.
    bool identity_busy(const std::string & client, const std::string & sid, const std::vector<bool> & active) const {
        if (!kvflash_client_explicit(client)) return false;
        for (size_t i = 0; i < slot_.size() && i < active.size(); ++i)
            if (active[i] && slot_[i].client == client && slot_[i].sid == sid) return true;
        return false;
    }
    bool durable(int slot, bool failed) const {
        const KvFlashSlotBundle & sb = slot_[(size_t)slot];
        return kvflash_session_park_durable(sb.pending, failed, sb.rewrite, sb.head.gen);
    }
    std::function<bool(const std::string &)> resume_claim;   // P3: may this disk session be resumed now? (unset: always)

    // The disk session a request must restore before it is admitted, if any.
    KvFlashRestorePlan plan_restore(const std::string & client, const std::string & sid, const std::vector<int32_t> & prompt, int e_role,
                                    const std::vector<bool> & active) const {
        KvFlashRestorePlan plan;
        if (identity_busy(client, sid, active)) { plan.pick = KvFlashRestorePick::busy; return plan; }
        const KvFlashSessionEntry * newest = store_.current(client, sid);
        if (newest && newest->state != "disk") return plan;   // the sid's current instance is in a slot (parked: the pick resumes it)
        const KvFlashSessionEntry * e = store_.pick_disk(client, sid, prompt, e_role);
        if (!e) return plan;
        plan.key = e->key;
        plan.pick = resume_claim && !resume_claim(plan.key) ? KvFlashRestorePick::busy : KvFlashRestorePick::load;
        return plan;
    }

private:
    std::vector<KvFlashSlotBundle> slot_;
    KvFlashSessionStore store_{std::string()};
    uint64_t identity_ = 0;
    bool persistent_ = false;
};

// ---- the engine's admission gate ----------------------------------------------------------------------

enum class KvFlashAdmitVerdict { proceed, invalid_identity, identity_busy, restore_busy };

// The checks that stand between a request and a slot, in the order the engine makes them: an identity the store cannot write down is
// refused; a second request for an identity that is running waits; a disk session that cannot be restored yet (no slot to leave, its
// storage claimed) waits too, instead of being admitted cold. `restore` is the engine's restore of a disk session (called only when
// the first two pass, and once). Pure over the book, so the engine's wiring is a call to this and a map of its verdict.
// `waiting`: a parked session holds a resume wait for this request (kvflash_pick_with_wait): the request resumes it, so no other
// disk session is restored first (each retry would otherwise read a bundle and evict again).
template <class Restore>
inline KvFlashAdmitVerdict kvflash_admit_gate(const KvFlashSessionBook & book, const KvFlashSessionId & who, const std::vector<bool> & active, bool waiting, Restore && restore) {
    if (!kvflash_identity_valid(who)) return KvFlashAdmitVerdict::invalid_identity;
    if (book.identity_busy(who.client, who.sid, active)) return KvFlashAdmitVerdict::identity_busy;
    return restore() == KvFlashRestore::busy ? KvFlashAdmitVerdict::restore_busy : KvFlashAdmitVerdict::proceed;
}
template <class Restore>
inline KvFlashAdmitVerdict kvflash_admit_gate(const KvFlashSessionBook & book, const KvFlashSessionId & who, const std::vector<bool> & active, Restore && restore) {
    return kvflash_admit_gate(book, who, active, false, std::forward<Restore>(restore));
}

}  // namespace luce::common
