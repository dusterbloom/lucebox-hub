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
//   DELETE /v1/sessions/{key}          {"incarnation"?: the token GET shows; another instance is a 412}; a repeat is 202 (running) or 204 (done)
#pragma once

#include "common/kvflash_storage.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <string_view>

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

// replace, never throw: an identity read from a foreign meta.json may not be UTF-8
inline SessionsApiReply reply(int status, const json & j) {
    return {status, status == 204 ? std::string() : j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n"};
}
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
    j["workers"] = {{"live", st.live_actions()}, {"stalled", st.stalled_actions()}};   // accepted storage actions unfinished; those past the deadline
    j["bytes"] = {{"live", o.live}, {"dead", o.dead}, {"total", o.total}, {"reserved_peaks", o.reserved_peaks}, {"quota", o.quota}, {"free", o.free}};
    return j;
}

inline json entry_json(const KvFlashSessionEntry & e, const std::string & incarnation) {
    const char * path = e.state == "parked-gpu" ? "resume" : e.state == "disk" ? "load" : e.state == "compact" ? "re-prefill" : nullptr;
    // ponytail: the restore estimates are flat rates (a disk load at 2 GB/s, a re-prefill at 775 tok/s, the R9700 figure of BYTES_PER_SESSION.md), labelled estimates by the key name
    const double est = e.state == "disk" ? (double)e.bytes_live / 2e9 : e.state == "compact" ? (double)e.tokens / 775.0 : 0.0;
    json restore = json::object();
    if (path) restore = {{"path", path}, {"est_s", est}};
    else restore = nullptr;
    return {{"key", e.key}, {"sid", e.sid}, {"client", e.client}, {"state", e.state}, {"slot", nullptr}, {"tokens", e.tokens},
            {"bytes_live", e.bytes_live}, {"bytes_dead", e.bytes_dead}, {"created", e.created}, {"last_used", e.last_used},
            {"archive_of", e.archive_of}, {"archive", e.archive}, {"gen", e.gen}, {"restore", restore}, {"incarnation", incarnation}};
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

// A session key: one word of [A-Za-z0-9_-], or a quarantine name slotN.quarantine.N the migration generated. Never a path.
inline bool key_ok(const std::string & k) {
    const auto word = [](std::string_view w) {
        return !w.empty() && std::all_of(w.begin(), w.end(), [](char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '-'; });
    };
    if (k.empty() || k.size() > 255) return false;
    const size_t q = k.find(".quarantine.");
    if (q == std::string::npos) return word(k);
    const std::string_view tail(k.data() + q + 12, k.size() - q - 12);
    return word(std::string_view(k.data(), q)) && !tail.empty() && std::all_of(tail.begin(), tail.end(), [](char c) { return c >= '0' && c <= '9'; });
}
// An optional field: absent is fine, present must be the right type. False with the 400 message in `err`.
inline bool opt_string(const json & b, const char * k, std::string & out, std::string & err) {
    const auto it = b.find(k);
    if (it == b.end()) return true;
    if (!it->is_string()) { err = std::string(k) + ": a string is required"; return false; }
    out = it->get<std::string>();
    return true;
}
inline bool opt_bool(const json & b, const char * k, bool & out, std::string & err) {
    const auto it = b.find(k);
    if (it == b.end()) return true;
    if (!it->is_boolean()) { err = std::string(k) + ": true or false is required"; return false; }
    out = it->get<bool>();
    return true;
}
inline bool body_object(const std::string & body, json & out) {
    if (body.empty()) { out = json::object(); return true; }
    out = json::parse(body, nullptr, false);
    return !out.is_discarded() && out.is_object();
}
}  // namespace sessions_api_detail

// `method` upper-case, `path` without the query. Runs on the engine's thread.
inline SessionsApiReply sessions_api_route_(KvFlashStorage & st, const std::string & method, const std::string & path, const std::string & body,
                                            const std::function<void(const std::string &)> & reserved) {
    using namespace sessions_api_detail;
    json b;
    std::string err, xpath_arg;
    if (path == "/v1/storage/policy") {
        if (method == "GET") return reply(200, policy_json(st));
        if (method == "PUT") {
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
        if (!opt_string(b, "path", xpath_arg, err)) return error(400, err);
        return from(st.answer(path.substr(20), b["action"].get<std::string>(), xpath_arg), 200, {{"ok", true}});
    }
    if (path == "/v1/sessions") {
        if (method != "GET") return error(405, "GET");
        json out = json::array();
        for (const auto & e : st.store().entries()) out.push_back(entry_json(e, st.store().incarnation(e.key)));
        return reply(200, out);
    }
    if (path == "/v1/sessions/import") {
        if (method != "POST") return error(405, "POST");
        if (!body_object(body, b) || !b.contains("path") || !b["path"].is_string()) return error(400, "path: the exported session directory is required");
        bool foreign = false;
        if (!opt_bool(b, "foreign", foreign, err)) return error(400, err);
        std::string key;
        const KvFlashResult r = st.import_dir(b["path"].get<std::string>(), foreign, &key, reserved);
        return from(r, 202, {{"key", key}});
    }
    if (path.rfind("/v1/sessions/", 0) != 0) return error(404, "unknown endpoint");
    std::string key = path.substr(13), verb;
    if (const size_t slash = key.find('/'); slash != std::string::npos) { verb = key.substr(slash + 1); key = key.substr(0, slash); }
    if (!key_ok(key)) return error(400, "key: a session key");
    if (verb.empty()) {
        if (method == "DELETE") {   // 202 when the removal runs on a worker (also for a repeat that joins it); 204 for a repeat of a finished one
            // optional {"incarnation": "<token>"} (GET shows it): the instance the caller means. A key is reused by the same session, so a
            // delayed retry of a delete that went through must not take the new instance with it.
            if (!body_object(body, b)) return error(400, "body: a JSON object");
            if (const auto it = b.find("incarnation"); it != b.end()) {
                if (!it->is_string()) return error(400, "incarnation: a string (the token GET shows) is required");
                if (st.store().find_key(key) && st.store().incarnation(key) != it->get<std::string>()) return error(412, "incarnation: the session under this key is another instance");
            }
            return from(st.remove(st.claim_of(key)), 204, {{"key", key}, {"state", "deleting"}});
        }
        if (method != "GET") return error(405, "GET or DELETE");
        const KvFlashSessionEntry * e = st.store().find_key(key);
        std::string xpath, xstate, xerr;
        const bool exported = st.export_result(key, xpath, xstate, &xerr);
        std::string istate, ierr;
        const bool importing = !e && st.import_result(key, istate, &ierr) && (istate == "importing" || istate == "failed");
        std::string op, ostate, oerr;
        const bool operated = st.op_result(key, op, ostate, &oerr);
        const bool gone_failed = !e && operated && ostate == "failed";   // a delete that removed the directory and could not make it durable
        if (!e && !exported && !importing && !gone_failed) return error(404, "no such session");
        json j = e ? entry_json(*e, st.store().incarnation(key)) : importing ? json{{"key", key}, {"state", istate}, {"error", ierr}} : exported ? json{{"key", key}, {"state", "exported"}} : json{{"key", key}, {"state", "deleted"}};
        KvFlashSlotHead h;
        if (e && st.read_head(key, h)) j["head"] = {{"gen", h.gen}, {"ckpt_pos", h.ckpt_pos}, {"trunk_pos", h.trunk_pos}, {"chunks", h.map.size()}};
        if (exported) j["export"] = {{"path", xpath}, {"state", xstate}, {"error", xerr}};
        if (operated) j["operation"] = {{"op", op}, {"state", ostate}, {"error", oerr}};   // a compact or delete on a worker: running | done | failed
        return reply(200, j);
    }
    if (method != "POST") return error(405, "POST");
    if (verb == "compact") return from(st.compact(st.claim_of(key)), 202, {{"key", key}, {"state", "compact"}});
    if (verb == "export") {
        if (!body_object(body, b)) return error(400, "body: a JSON object");
        if (!opt_string(b, "path", xpath_arg, err)) return error(400, err);
        return from(st.export_async(st.claim_of(key), xpath_arg), 202, {{"key", key}});
    }
    return error(404, "unknown endpoint");
}

// No exception leaves a request: it is that request's 400.
// `reserved` hears the key of an import as soon as it is held (the caller's deadline answer can name it).
inline SessionsApiReply sessions_api_route(KvFlashStorage & st, const std::string & method, const std::string & path, const std::string & body,
                                           const std::function<void(const std::string &)> & reserved = {}) {
    try {
        return sessions_api_route_(st, method, path, body, reserved);
    } catch (const std::exception & e) {
        return sessions_api_detail::error(400, std::string("request failed: ") + e.what());
    } catch (...) {
        return sessions_api_detail::error(400, "request failed");
    }
}

}  // namespace luce::common
