// KVFlash 2.0 sessions P3 (trackc/SESSIONS_DESIGN.md section 6): the storage and session routes as a pure
// function over a KvFlashStorage. The HTTP thread never calls it: HttpServer queues the request to the
// scheduler thread, which runs it against the engine's storage (the only thread that touches the store).
//
//   GET  /v1/storage/policy            PUT /v1/storage/policy
//   GET  /v1/storage/pending           POST /v1/storage/pending/{id}      {"action": "compact|export|delete", "path"?}
//   GET  /v1/storage/preambles         (P1's store; empty here)
//   GET  /v1/sessions                  GET /v1/sessions/{key}
//   POST /v1/sessions/{key}/compact    POST /v1/sessions/{key}/export     {"path"?}
//   POST /v1/sessions/import           {"path", "foreign"?}
//   DELETE /v1/sessions/{key}
#pragma once

#include "common/kvflash_storage.h"

#include <nlohmann/json.hpp>

#include <string>

namespace luce::common {

struct SessionsApiReply {
    int status = 404;
    std::string body;
};

inline bool sessions_api_owns(const std::string & path) {
    return path == "/v1/sessions" || path.rfind("/v1/sessions/", 0) == 0 || path.rfind("/v1/storage/", 0) == 0;
}

namespace sessions_api_detail {
using json = nlohmann::json;

inline SessionsApiReply reply(int status, const json & j) { return {status, status == 204 ? std::string() : j.dump() + "\n"}; }
inline SessionsApiReply error(int status, const std::string & msg) {
    json j = {{"error", {{"message", msg}}}};
    const size_t colon = msg.find(':');
    if (status == 400 && colon != std::string::npos && msg.find(' ') > colon) j["error"]["field"] = msg.substr(0, colon);   // "quota_gib: ..." names its field
    return reply(status, j);
}
inline SessionsApiReply from(const KvFlashResult & r, int ok_status, json ok_body = json::object()) {
    if (!r.ok()) return error(r.status, r.error);
    if (!r.error.empty()) ok_body["warning"] = r.error;   // a committed compaction with an unlink still owed
    return reply(r.status == 202 ? 202 : ok_status, ok_body);
}

inline json policy_json(KvFlashStorage & st) {
    json j = json::parse(kvflash_policy_pack(st.policy()));
    const KvFlashOccupancy o = st.occupancy();
    j["dir"] = st.store().root();
    j["bytes"] = {{"live", o.live}, {"dead", o.dead}, {"total", o.total}, {"reserved_peaks", o.reserved_peaks}, {"quota", o.quota}, {"free", o.free}};
    return j;
}

inline json entry_json(const KvFlashSessionEntry & e) {
    const char * path = e.state == "parked-gpu" ? "resume" : e.state == "disk" ? "load" : e.state == "compact" ? "re-prefill" : nullptr;
    // ponytail: the restore estimates are flat rates (a disk load at 2 GB/s, a re-prefill at 775 tok/s, the R9700 figure of BYTES_PER_SESSION.md), labelled estimates by the key name
    const double est = e.state == "disk" ? (double)e.bytes_live / 2e9 : e.state == "compact" ? (double)e.tokens / 775.0 : 0.0;
    json restore = json::object();
    if (path) restore = {{"path", path}, {"est_s", est}};
    else restore = nullptr;
    return {{"key", e.key}, {"sid", e.sid}, {"client", e.client}, {"state", e.state}, {"slot", nullptr}, {"tokens", e.tokens},
            {"bytes_live", e.bytes_live}, {"bytes_dead", e.bytes_dead}, {"created", e.created}, {"last_used", e.last_used},
            {"archive_of", e.archive_of}, {"archive", e.archive}, {"gen", e.gen}, {"restore", restore}};
}

inline json pending_json(KvFlashStorage & st) {
    json out = json::array();
    for (const KvFlashPending & p : st.pending()) {
        const KvFlashSessionEntry * e = st.store().find_key(p.key);
        const bool has_compact = std::find(p.options.begin(), p.options.end(), "compact") != p.options.end();
        out.push_back({{"id", p.id}, {"key", p.key}, {"sid", e ? e->sid : ""}, {"client", e ? e->client : ""}, {"gen", p.gen}, {"reason", p.reason},
                       {"tokens", e ? e->tokens : 0}, {"bytes", e ? e->bytes_live : 0}, {"last_used", p.last_used}, {"deadline", p.deadline},
                       {"options", p.options}, {"default", has_compact ? json("compact") : json(nullptr)}});
    }
    return out;
}

inline bool body_object(const std::string & body, json & out) {
    if (body.empty()) { out = json::object(); return true; }
    out = json::parse(body, nullptr, false);
    return !out.is_discarded() && out.is_object();
}
}  // namespace sessions_api_detail

// `method` upper-case, `path` without the query. Runs on the engine's thread.
inline SessionsApiReply sessions_api_route(KvFlashStorage & st, const std::string & method, const std::string & path, const std::string & body) {
    using namespace sessions_api_detail;
    json b;
    if (path == "/v1/storage/policy") {
        if (method == "GET") return reply(200, policy_json(st));
        if (method == "PUT") {
            std::string err;
            if (!st.set_policy(body, err)) return error(400, err);
            return reply(200, policy_json(st));
        }
        return error(405, "GET or PUT");
    }
    if (path == "/v1/storage/pending") return method == "GET" ? reply(200, pending_json(st)) : error(405, "GET");
    if (path == "/v1/storage/preambles") return method == "GET" ? reply(200, json::array()) : error(405, "GET");
    if (path.rfind("/v1/storage/pending/", 0) == 0) {
        if (method != "POST") return error(405, "POST");
        if (!body_object(body, b) || !b.contains("action") || !b["action"].is_string()) return error(400, "action: compact, export or delete");
        return from(st.answer(path.substr(20), b["action"], b.value("path", std::string())), 200, {{"ok", true}});
    }
    if (path == "/v1/sessions") {
        if (method != "GET") return error(405, "GET");
        json out = json::array();
        for (const auto & e : st.store().entries()) out.push_back(entry_json(e));
        return reply(200, out);
    }
    if (path == "/v1/sessions/import") {
        if (method != "POST") return error(405, "POST");
        if (!body_object(body, b) || !b.contains("path") || !b["path"].is_string()) return error(400, "path: the exported session directory is required");
        std::string key;
        const KvFlashResult r = st.import_dir(b["path"], b.value("foreign", false), &key);
        return from(r, 202, {{"key", key}});
    }
    if (path.rfind("/v1/sessions/", 0) != 0) return error(404, "unknown endpoint");
    std::string key = path.substr(13), verb;
    if (const size_t slash = key.find('/'); slash != std::string::npos) { verb = key.substr(slash + 1); key = key.substr(0, slash); }
    if (key.empty() || key.find_first_of("./\\") != std::string::npos) return error(400, "key: a session key");
    if (verb.empty()) {
        if (method == "DELETE") return from(st.remove(st.claim_of(key)), 204);
        if (method != "GET") return error(405, "GET or DELETE");
        const KvFlashSessionEntry * e = st.store().find_key(key);
        std::string xpath, xstate, xerr;
        const bool exported = st.export_result(key, xpath, xstate, &xerr);
        if (!e && !exported) return error(404, "no such session");
        json j = e ? entry_json(*e) : json{{"key", key}, {"state", "exported"}};
        KvFlashSlotHead h;
        if (e && st.read_head(key, h)) j["head"] = {{"gen", h.gen}, {"ckpt_pos", h.ckpt_pos}, {"trunk_pos", h.trunk_pos}, {"chunks", h.map.size()}};
        if (exported) j["export"] = {{"path", xpath}, {"state", xstate}, {"error", xerr}};
        return reply(200, j);
    }
    if (method != "POST") return error(405, "POST");
    if (verb == "compact") return from(st.compact(st.claim_of(key)), 202, {{"key", key}, {"state", "compact"}});
    if (verb == "export") {
        if (!body_object(body, b)) return error(400, "body: a JSON object");
        return from(st.export_async(st.claim_of(key), b.value("path", std::string())), 202, {{"key", key}});
    }
    return error(404, "unknown endpoint");
}

}  // namespace luce::common
