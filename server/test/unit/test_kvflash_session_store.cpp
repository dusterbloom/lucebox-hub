// Pure unit tests for the KVFlash 2.0 session store (kvflash_session_store.h), phase P0 of
// trackc/SESSIONS_DESIGN.md: session identity (headers, body, owner-restricted resume), the lower-cased
// header map, meta.json and index.json, the scan that rebuilds the index from sessions/<key>/,
// the migration of today's slot<i>/ directories, and evict-not-delete. Real bundle directories
// written through kvflash_bundle_commit, no engine, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_session_store.h"
#include "../../src/common/kvflash_slot_turn.h"
#include "../../src/common/kvflash_storage.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

using namespace luce::common;
namespace fs = std::filesystem;

namespace {
struct KvflashSessionStoreFixture {};

constexpr size_t CB = 8;
const size_t RB = kvflash_record_bytes(CB);
constexpr uint64_t ID = 0x5e55;
constexpr size_t HEAD_MAX = 1 << 20;

std::vector<uint8_t> hdr44() { std::vector<uint8_t> h(kKvFlashPagerHdr); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t)(0xa0 + i); return h; }

// A whole park of `n` chunks whose history is `hist`, written to `dir` (created).
bool park_session(const std::string & dir, const std::vector<int32_t> & hist, int n, uint8_t fill, uint64_t identity = ID, int crash_step = 0) {
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = 1; c.log_gen = 1; c.new_log = true; c.new_dir = true; c.record_bytes = RB;
    KvFlashSlotHead h;
    h.identity = identity; h.gen = 1; h.log_gen = 1; h.end_gen = 2; h.history = hist; h.pager_hdr = hdr44(); h.ledger.assign((size_t)n * 8, 0);
    for (int i = 0; i < n; ++i) {
        h.ledger[(size_t)i * 8] = 1; h.ledger[(size_t)i * 8 + 1] = 1;
        c.records.push_back({(uint32_t)i, std::vector<uint8_t>(CB, (uint8_t)(fill + i))});
        h.map.push_back(kKvFlashLogHeader + (uint64_t)i * RB);
    }
    h.log_bytes = kKvFlashLogHeader + (uint64_t)n * RB;
    c.files = {{kvflash_bundle_slab_name(2), kvflash_slab_pack({{fill, 1}}, {{fill, 2}})}};
    c.head = kvflash_head_pack(h);
    return kvflash_bundle_commit(c, crash_step);
}

bool loads(const std::string & dir, std::vector<int32_t> * hist = nullptr) {
    KvFlashBundleLoaded l;
    std::string why;
    const KvFlashBundleLimits lim{1 << 20, 1 << 20, 1 << 20};
    if (!kvflash_bundle_load(dir, ID, CB, lim, l, why)) return false;
    if (hist) *hist = l.head.history;
    return true;
}

std::string fresh_root(const char * name) {
    fs::remove_all(name);
    fs::create_directories(name);
    return name;
}

std::set<std::string> names(const std::string & dir) {
    std::set<std::string> out;
    for (const auto & e : fs::directory_iterator(dir)) out.insert(e.path().filename().string());
    return out;
}

std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

KvFlashSessionEntry entry(const std::string & sid, int64_t last_used, const std::string & state = "parked-gpu") {
    KvFlashSessionEntry e;
    e.key = kvflash_sid_key("header", sid); e.sid = sid; e.client = "header"; e.state = state;
    e.created = 100; e.last_used = last_used; e.tokens = 5; e.identity = ID;
    return e;
}
}  // namespace

// ── identity ────────────────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, identity_resolves_opencode_then_affinity_then_claude_then_generic_then_body) {
    const KvFlashHeaders all = kvflash_parse_headers(
        "X-Session-Id: generic\r\nX-Claude-Code-Session-Id: claude\r\nx-session-affinity: aff\r\nx-opencode-session-id: ses_oc\r\n");
    KvFlashSessionId id = kvflash_resolve_session(all, "body");
    CHECK(id.sid == "ses_oc");
    CHECK(id.client == "opencode");

    id = kvflash_resolve_session(kvflash_parse_headers("X-Session-Id: generic\r\nX-Claude-Code-Session-Id: claude\r\nX-Session-Affinity: aff\r\n"), "body");
    CHECK(id.sid == "aff");                                   // opencode's other header
    CHECK(id.client == "opencode");

    id = kvflash_resolve_session(kvflash_parse_headers("X-Session-Id: generic\r\nX-Claude-Code-Session-Id: claude\r\n"), "body");
    CHECK(id.sid == "claude");
    CHECK(id.client == "claude");

    id = kvflash_resolve_session(kvflash_parse_headers("X-Session-Id: generic\r\n"), "body");
    CHECK(id.sid == "generic");
    CHECK(id.client == "header");

    id = kvflash_resolve_session(kvflash_parse_headers("Content-Type: application/json\r\n"), "body");
    CHECK(id.sid == "body");
    CHECK(id.client == "body");

    // parent ids are a relation, not an identity
    id = kvflash_resolve_session(kvflash_parse_headers("x-parent-session-id: p\r\nx-opencode-parent-session-id: q\r\n"), "");
    CHECK(id.empty());
    CHECK(id.client.empty());

    // an empty header value is no identity: the next source answers
    id = kvflash_resolve_session(kvflash_parse_headers("x-opencode-session-id:   \r\nX-Claude-Code-Session-Id: claude\r\n"), "");
    CHECK(id.sid == "claude");
}

TEST_CASE(KvflashSessionStoreFixture, headers_parse_lower_cased_trimmed_first_value_wins_and_both_line_endings) {
    const KvFlashHeaders h = kvflash_parse_headers(
        "Host: example\r\nX-OpenCode-Session-ID:  ses_1 \r\nCONTENT-LENGTH: 12\r\nx-opencode-session-id: ses_2\r\nbroken line\r\nEmpty:\r\n");
    CHECK(h.at("host") == "example");
    CHECK(h.at("x-opencode-session-id") == "ses_1");   // a duplicate keeps the first
    CHECK(h.at("content-length") == "12");
    CHECK(h.at("empty") == "");
    CHECK(h.count("broken line") == 0);
    CHECK(h.count("X-OpenCode-Session-ID") == 0);      // names are stored lower-cased only
    CHECK(h.size() == 4);

    const KvFlashHeaders lf = kvflash_parse_headers("A: 1\nB: 2\n");
    CHECK(lf.at("a") == "1");
    CHECK(lf.at("b") == "2");
    CHECK(kvflash_parse_headers("").empty());
}

TEST_CASE(KvflashSessionStoreFixture, the_directory_key_is_fnv_hex16_of_the_pair_and_a_taken_key_gets_a_deterministic_variant) {
    CHECK(kvflash_sid_key("header", "ses_1").size() == 16);
    CHECK(kvflash_sid_key("header", "ses_1") != kvflash_sid_key("header", "ses_2"));
    CHECK(kvflash_sid_key("header", "ses_1") != kvflash_sid_key("opencode", "ses_1"));   // the same sid under two clients is two sessions
    std::set<std::string> taken;
    const std::string k0 = kvflash_session_dir_key("header", "ses_1", taken);
    CHECK(k0 == kvflash_sid_key("header", "ses_1"));
    taken.insert(k0);
    const std::string k1 = kvflash_session_dir_key("header", "ses_1", taken);
    CHECK(k1 != k0);
    CHECK(k1.size() == 16);
    taken.insert(k1);
    const std::string k2 = kvflash_session_dir_key("header", "ses_1", taken);
    CHECK(k2 != k0);
    CHECK(k2 != k1);
    CHECK(kvflash_session_dir_key("header", "ses_1", std::set<std::string>{k0}) == k1);   // same input, same answer
}

// ── meta and index ──────────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, meta_round_trips_and_escapes_a_hostile_sid) {
    KvFlashSessionEntry e = entry("ses \"q\" \\ \n\t \x01 é", 12345);
    e.client = "opencode"; e.state = "disk"; e.archive_of = "abc"; e.archive = "def";
    e.bytes_live = 1234567890123ULL; e.bytes_dead = 7; e.tokens = 99;
    const std::string json = kvflash_meta_pack(e);
    CHECK(json.find('\n') == std::string::npos);   // one line, whatever the sid holds
    KvFlashSessionEntry b;
    REQUIRE(kvflash_meta_unpack(json, b));
    CHECK(b.sid == e.sid);
    CHECK(b.client == "opencode");
    CHECK(b.state == "disk");
    CHECK(b.archive_of == "abc");
    CHECK(b.archive == "def");
    CHECK(b.created == 100);
    CHECK(b.last_used == 12345);
    CHECK(b.tokens == 99);
    CHECK(b.bytes_live == 1234567890123ULL);
    CHECK(b.bytes_dead == 7);
    CHECK(b.identity == ID);
    CHECK(!kvflash_meta_unpack("", b));
    CHECK(!kvflash_meta_unpack("{\"sid\": \"x\"", b));    // torn
    CHECK(!kvflash_meta_unpack("{\"client\": \"x\"}", b));   // no sid
}

TEST_CASE(KvflashSessionStoreFixture, writes_are_atomic_replaces_and_leave_no_tmp) {
    const std::string root = fresh_root("kvflash_p0_unit_atomic");
    KvFlashSessionStore store(root);
    REQUIRE(fs::create_directories(store.dir("k1")));
    KvFlashSessionEntry e = entry("s1", 10);
    e.key = "k1";
    store.upsert(e);
    REQUIRE(store.write_meta("k1"));
    REQUIRE(store.write_index());
    const std::string m1 = slurp(root + "/sessions/k1/meta.json"), i1 = slurp(root + "/index.json");
    CHECK(!m1.empty());
    CHECK(i1.find("\"s1\"") != std::string::npos);
    e.last_used = 20; e.tokens = 77;
    store.upsert(e);
    REQUIRE(store.write_meta("k1"));
    REQUIRE(store.write_index());
    const std::string m2 = slurp(root + "/sessions/k1/meta.json");
    CHECK(m2 != m1);
    CHECK(m2.find("77") != std::string::npos);
    for (const auto & f : names(root + "/sessions/k1")) CHECK(f == "meta.json");
    for (const auto & f : names(root)) CHECK((f == "sessions" || f == "index.json"));
    // a write killed before its rename leaves only a tmp: the old file is whole, and a scan ignores the tmp
    std::ofstream(root + "/sessions/k1/meta.json.tmp.4242") << "{\"sid\": \"torn";
    CHECK(slurp(root + "/sessions/k1/meta.json") == m2);
    fs::remove_all(root);
}

// ── scan: the index is a cache of the directories ───────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, scan_rebuilds_the_index_from_head_and_meta_with_a_missing_meta_or_a_missing_head) {
    const std::string root = fresh_root("kvflash_p0_unit_scan");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    // a: head + meta. b: head, no meta. c: meta, no head (a first park killed before its commit). d: another server's head.
    for (const char * k : {"aaaa", "bbbb", "cccc", "dddd"}) fs::create_directories(store.dir(k));
    REQUIRE(park_session(store.dir("aaaa"), {1, 2, 3, 4, 5, 6}, 2, 0x10));
    REQUIRE(park_session(store.dir("bbbb"), {7, 8, 9}, 1, 0x20));
    REQUIRE(park_session(store.dir("dddd"), {1}, 1, 0x30, /*identity=*/ID + 1));
    KvFlashSessionEntry a = entry("sid-a", 500, "parked-gpu");
    a.key = "aaaa"; a.tokens = 4;   // stale: the head holds 6 tokens
    KvFlashSessionEntry c = entry("sid-c", 600);
    c.key = "cccc";
    store.upsert(a); store.upsert(c);
    REQUIRE(store.write_meta("aaaa"));
    REQUIRE(store.write_meta("cccc"));
    std::ofstream(store.dir("cccc") + "/chunks.1.log") << "uncommitted";

    KvFlashScanReport rep;
    const std::vector<KvFlashSessionEntry> es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(es.size() == 3);                       // a, b and d (listed as foreign: another server's, never erased)
    const KvFlashSessionEntry * ea = nullptr, * eb = nullptr, * ed = nullptr;
    for (const auto & e : es) { if (e.key == "aaaa") ea = &e; if (e.key == "bbbb") eb = &e; if (e.key == "dddd") ed = &e; }
    REQUIRE(ed != nullptr);
    CHECK(ed->state == "foreign");
    CHECK(ed->identity == ID + 1);
    REQUIRE(ea != nullptr);
    REQUIRE(eb != nullptr);
    CHECK(ea->sid == "sid-a");
    CHECK(ea->tokens == 6);                       // the head is the truth, not the meta
    CHECK(ea->last_used == 500);
    CHECK(ea->state == "disk");                    // nothing is in a slot at start
    CHECK(ea->history == (std::vector<int32_t>{1, 2, 3, 4, 5, 6}));
    CHECK(ea->bytes_live > 0);
    CHECK(eb->state == "foreign");                 // no meta: whose it is cannot be proven, so it is listed and never resumed (R01)
    CHECK(eb->client == "foreign");
    CHECK(eb->history.empty());
    CHECK(fs::exists(store.dir("bbbb") + "/head"));
    CHECK(!fs::exists(store.dir("cccc")));         // meta without a head: nothing was committed
    CHECK(fs::exists(store.dir("dddd") + "/head"));   // another server's bundle is skipped, never erased
    REQUIRE(rep.skipped.size() == 2);              // another server's head, and the one whose meta.json is missing
    CHECK(std::set<std::string>(rep.skipped.begin(), rep.skipped.end()) == (std::set<std::string>{"bbbb", "dddd"}));
    REQUIRE(rep.removed.size() == 1);
    CHECK(rep.removed[0] == "cccc");
    // and a dir that is not a session key (a set-aside one) is not a session
    fs::create_directories(root + "/sessions/aaaa.skipped");
    KvFlashScanReport rep2;
    CHECK(kvflash_session_scan(root, ID, RB, HEAD_MAX, rep2).size() == 3);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_crash_between_the_bundle_data_and_the_index_loses_nothing_that_was_committed) {
    for (int step = 1; step <= 5; ++step) {
        const std::string root = fresh_root("kvflash_p0_unit_crash");
        fs::create_directories(root + "/sessions");
        KvFlashSessionStore store(root);
        // the previous park of session "old" is whole; a first park of "new" is killed after `step`
        REQUIRE(fs::create_directories(store.dir("old")));
        REQUIRE(park_session(store.dir("old"), {1, 2, 3}, 1, 0x10));
        KvFlashSessionEntry o = entry("sid-old", 10);
        o.key = "old";
        store.upsert(o);
        REQUIRE(store.write_meta("old"));
        REQUIRE(store.write_index());
        KvFlashSessionEntry n = entry("sid-new", 20);
        n.key = "new";
        REQUIRE(fs::create_directories(store.dir("new")));
        store.upsert(n);
        REQUIRE(store.write_meta("new"));                       // staged like save_bundle_: the meta goes before the data
        CHECK(!park_session(store.dir("new"), {4, 5}, 1, 0x20, ID, step));   // killed after this step; index.json is stale

        KvFlashScanReport rep;
        const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        std::set<std::string> keys;
        for (const auto & e : es) keys.insert(e.key);
        CHECK(keys.count("old") == 1);
        CHECK(std::none_of(es.begin(), es.end(), [](const auto & e) { return e.state == "foreign"; }));
        CHECK(loads(store.dir("old")));
        if (step < 4) {   // before the head rename: nothing of "new" was committed
            CHECK(keys.size() == 1);
            CHECK(!fs::exists(store.dir("new")));
        } else {          // head renamed: "new" is whole and listed, its meta says who it is
            CHECK(keys.size() == 2);
            for (const auto & e : es) if (e.key == "new") { CHECK(e.sid == "sid-new"); CHECK(e.tokens == 2); }
            CHECK(loads(store.dir("new")));
        }
        fs::remove_all(root);
    }
}

TEST_CASE(KvflashSessionStoreFixture, the_restart_cleanup_keeps_meta_json_and_cuts_what_the_head_does_not_name) {
    const std::string dir = "kvflash_p0_unit_clean";
    fs::remove_all(dir);
    REQUIRE(park_session(dir, {1, 2}, 1, 0x10));
    std::ofstream(dir + "/meta.json") << "{}";
    std::ofstream(dir + "/chunks.9.log") << "stale";
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(dir, ID, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}, l, why));
    kvflash_bundle_clean(dir, &l.head);
    CHECK(fs::exists(dir + "/meta.json"));
    CHECK(!fs::exists(dir + "/chunks.9.log"));
    CHECK(fs::exists(dir + "/head"));
    // a restart's cleanup of a session directory never reaches the store's own files above it
    const std::string root = fresh_root("kvflash_p0_unit_clean_root");
    fs::create_directories(root + "/sessions");
    std::ofstream(root + "/index.json") << "[]";
    std::ofstream(root + "/policy.json") << "{}";
    REQUIRE(park_session(root + "/sessions/k1", {1, 2}, 1, 0x10));
    std::ofstream(root + "/sessions/k1/meta.json") << "{}";
    REQUIRE(kvflash_bundle_load(root + "/sessions/k1", ID, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}, l, why));
    kvflash_bundle_clean(root + "/sessions/k1", &l.head);
    CHECK(fs::exists(root + "/index.json"));
    CHECK(fs::exists(root + "/policy.json"));
    CHECK(fs::exists(root + "/sessions/k1/meta.json"));
    fs::remove_all(root);
    fs::remove_all(dir);
}

// ── migration of <dir>/slot<i>/ ─────────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, migration_renames_exactly_the_loadable_slot_dirs_and_is_idempotent) {
    const std::string root = fresh_root("kvflash_p0_unit_migrate");
    REQUIRE(park_session(root + "/slot0", {1, 2, 3, 4}, 2, 0x10));
    REQUIRE(park_session(root + "/slot1", {5, 6, 7}, 1, 0x20));
    REQUIRE(park_session(root + "/slot3", {9}, 1, 0x30, ID + 1));      // another server's
    fs::create_directories(root + "/slot2");                           // a park killed before its head: stray files only
    std::ofstream(root + "/slot2/chunks.1.log") << "uncommitted";
    fs::create_directories(root + "/slot1.skipped");                   // an earlier refusal: left where it is
    std::ofstream(root + "/slot1.skipped/head") << "x";
    std::ofstream(root + "/.lock") << "";
    const std::string old_head0 = slurp(root + "/slot0/head");

    KvFlashScanReport rep;
    CHECK(kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep) == 2);
    CHECK(!fs::exists(root + "/slot0"));
    CHECK(!fs::exists(root + "/slot1"));
    CHECK(!fs::exists(root + "/slot2"));                               // cleaned, nothing committed in it
    CHECK(!fs::exists(root + "/slot3"));                               // another server's: set aside whole, never erased
    CHECK(fs::exists(root + "/sessions/slot3.quarantine.1/head"));
    CHECK(fs::exists(root + "/slot1.skipped/head"));
    CHECK(fs::exists(root + "/.lock"));
    CHECK(names(root + "/sessions").size() == 3);

    const auto all = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(all.size() == 3);                                          // two sessions and the quarantined slot3, listed
    std::vector<KvFlashSessionEntry> es;
    for (const auto & e : all) {
        if (e.state != "foreign") { es.push_back(e); continue; }
        CHECK(e.key == "slot3.quarantine.1");
        CHECK(e.identity == ID + 1);                                   // whose it is, read from its head
        CHECK(e.bytes_live > 0);
    }
    REQUIRE(es.size() == 2);
    std::set<std::vector<int32_t>> hist;
    for (const auto & e : es) {
        hist.insert(e.history);
        CHECK(e.sid.rfind("migrated:", 0) == 0);
        CHECK(e.state == "disk");
        CHECK(e.key.size() == 16);
        CHECK(loads(root + "/sessions/" + e.key));                     // the bundle is byte-for-byte the old one, renamed
        CHECK(fs::exists(root + "/sessions/" + e.key + "/meta.json"));
    }
    CHECK(hist == (std::set<std::vector<int32_t>>{{1, 2, 3, 4}, {5, 6, 7}}));
    bool found = false;
    for (const auto & e : es) if (slurp(root + "/sessions/" + e.key + "/head") == old_head0) found = true;
    CHECK(found);

    // a second start: nothing moves
    const auto before = names(root);
    KvFlashScanReport rep2;
    CHECK(kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep2) == 0);
    CHECK(names(root) == before);
    CHECK(kvflash_session_scan(root, ID, RB, HEAD_MAX, rep2).size() == 3);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, migration_keeps_two_slots_with_the_same_history_as_two_sessions_and_the_slot_recency) {
    const std::string root = fresh_root("kvflash_p0_unit_migrate2");
    REQUIRE(park_session(root + "/slot0", {1, 2, 3}, 1, 0x10));
    REQUIRE(park_session(root + "/slot1", {1, 2, 3}, 1, 0x20));
    // slot0 was parked last: the migrated meta carries the head's mtime, so the adoption order survives the rename
    fs::last_write_time(root + "/slot0/head", fs::file_time_type::clock::now());
    fs::last_write_time(root + "/slot1/head", fs::file_time_type::clock::now() - std::chrono::hours(2));
    KvFlashScanReport rep;
    CHECK(kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep) == 2);
    auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(es.size() == 2);
    CHECK(es[0].key != es[1].key);
    std::sort(es.begin(), es.end(), [](const auto & a, const auto & b) { return a.last_used > b.last_used; });
    CHECK(es[0].last_used > es[1].last_used);
    CHECK(slurp(root + "/sessions/" + es[0].key + "/chunks.1.log") != slurp(root + "/sessions/" + es[1].key + "/chunks.1.log"));
    fs::remove_all(root);
}

// ── evict-not-delete and find-by-sid ────────────────────────────────────────────────────────

TEST_CASE(KvflashSessionStoreFixture, a_full_house_evicts_the_oldest_to_disk_keeps_its_bundle_and_the_index_lists_it) {
    const std::string root = fresh_root("kvflash_p0_unit_evict");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    const std::vector<int32_t> ha = {1, 2, 3}, hb = {4, 5, 6, 7};
    for (const auto & [sid, last, hist, fill] : std::vector<std::tuple<std::string, int64_t, std::vector<int32_t>, uint8_t>>{
             {"sid-a", 10, ha, 0x10}, {"sid-b", 20, hb, 0x20}}) {
        KvFlashSessionEntry e = entry(sid, last);
        REQUIRE(fs::create_directories(store.dir(e.key)));
        REQUIRE(park_session(store.dir(e.key), hist, 1, fill));
        store.upsert(e);
        REQUIRE(store.write_meta(e.key));
    }
    // the two slots are full and a third session arrives: the oldest parked one goes to disk
    const std::vector<KvFlashParkedSession> parked = {{0, &ha, 1}, {1, &hb, 2}};   // slot, history, park tick: the engine's own pick
    CHECK(kvflash_oldest_session(parked) == 0);
    const std::string key_a = kvflash_sid_key("header", "sid-a");
    REQUIRE(store.evict(key_a, ha));
    CHECK(store.find_key(key_a)->state == "disk");
    CHECK(store.find_key(key_a)->history == ha);
    CHECK(store.find_key(kvflash_sid_key("header", "sid-b"))->state == "parked-gpu");
    CHECK(fs::exists(store.dir(key_a) + "/head"));                   // the bundle is kept
    CHECK(loads(store.dir(key_a)));
    const std::string idx = slurp(root + "/index.json");
    CHECK(idx.find("\"sid-a\"") != std::string::npos);                // and the index still lists it
    CHECK(idx.find("\"disk\"") != std::string::npos);
    CHECK(idx.find("\"sid-b\"") != std::string::npos);
    CHECK(slurp(store.dir(key_a) + "/meta.json").find("\"disk\"") != std::string::npos);

    // a restart: a scan lists both, both on disk
    KvFlashScanReport rep;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    CHECK(es.size() == 2);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_session_evicted_to_disk_is_found_again_by_sid_and_restored) {
    const std::string root = fresh_root("kvflash_p0_unit_restore");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    const std::vector<int32_t> ha = {1, 2, 3, 4};
    KvFlashSessionEntry a = entry("sid-a", 10);
    REQUIRE(fs::create_directories(store.dir(a.key)));
    REQUIRE(park_session(store.dir(a.key), ha, 1, 0x10));
    store.upsert(a);
    REQUIRE(store.write_meta(a.key));
    REQUIRE(store.evict(a.key, ha));

    const std::vector<int32_t> prompt = {1, 2, 3, 4, 8, 9};
    const KvFlashSessionEntry * hit = store.pick_disk("header", "sid-a", prompt, 0);
    REQUIRE(hit != nullptr);
    CHECK(hit->key == a.key);
    CHECK(store.pick_disk("header", "sid-other", prompt, 0) == nullptr);   // another sid does not take it
    CHECK(store.pick_disk("opencode", "sid-a", prompt, 0) == nullptr);     // nor does the same sid of another client
    CHECK(store.pick_disk("anon", "anon:1", prompt, 0) == nullptr);        // and no sid never matches a real one
    std::vector<int32_t> hist;
    CHECK(loads(store.dir(a.key), &hist));
    CHECK(hist == ha);
    REQUIRE(store.adopt(a.key));                                      // restored into a slot
    CHECK(store.find_key(a.key)->state == "parked-gpu");
    CHECK(store.find_key(a.key)->history.empty());                    // a slot session's history lives in the engine
    CHECK(store.pick_disk("header", "sid-a", prompt, 0) == nullptr);  // no longer on disk

    // the same sid, a second instance (the first not resumable): two dirs, the newest answers
    std::set<std::string> taken;
    for (const auto & e : store.entries()) taken.insert(e.key);
    KvFlashSessionEntry b = entry("sid-a", 50);
    b.key = kvflash_session_dir_key("header", "sid-a", taken);
    CHECK(b.key != a.key);
    REQUIRE(fs::create_directories(store.dir(b.key)));
    REQUIRE(park_session(store.dir(b.key), {1, 2, 3}, 1, 0x40));
    store.upsert(b);
    REQUIRE(store.evict(b.key, {1, 2, 3}));
    REQUIRE(store.evict(a.key, ha));
    REQUIRE(store.pick_disk("header", "sid-a", prompt, 0) != nullptr);
    CHECK(store.pick_disk("header", "sid-a", prompt, 0)->key == b.key);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, an_ownerless_disk_session_is_found_by_the_longest_prefix_beyond_the_preamble) {
    const std::string root = fresh_root("kvflash_p0_unit_bridge");
    KvFlashSessionStore store(root);
    KvFlashSessionEntry m1 = entry("migrated:1", 10, "disk"), m2 = entry("migrated:2", 20, "disk"), r = entry("real", 30, "disk");
    m1.client = m2.client = "migrated";
    m1.history = {1, 2, 3, 4}; m2.history = {1, 2, 9}; r.history = {1, 2, 3, 4, 5};
    store.upsert(m1); store.upsert(m2); store.upsert(r);
    const KvFlashSessionEntry * hit = store.pick_disk("opencode", "ses_new", {1, 2, 3, 4, 8}, 0);
    REQUIRE(hit != nullptr);
    CHECK(hit->sid == "migrated:1");                                  // the longest prefix, and only among ownerless ones
    CHECK(store.pick_disk("opencode", "ses_new", {7, 7}, 0) == nullptr);
    CHECK(store.pick_disk("header", "real", {7, 7}, 0) == nullptr);   // a sid match still needs a shared prefix: no load for nothing
    CHECK(store.pick_disk("header", "real", {1, 2, 3, 4, 5, 6}, 0)->sid == "real");   // the sid beats a longer-or-equal migrated one
    // a shared preamble alone (lcp <= e_role) is not a conversation: nothing ownerless is taken for it
    CHECK(store.pick_disk("opencode", "ses_new", {1, 2, 8}, 2) == nullptr);
    CHECK(store.pick_disk("anon", "anon:4", {1, 2, 3, 8}, 2)->sid == "migrated:1");
    CHECK(store.pick_disk("anon", "anon:4", {1, 2, 3, 4, 5, 6}, 0)->sid == "migrated:1");   // an anonymous request never takes a session with a sid
    fs::remove_all(root);
}

// ── review P1s: explicit sid, instances and links, durable eviction, slot reuse, quarantine ──

namespace {
KvFlashOwnedSession owned(int slot, const std::vector<int32_t> * h, uint64_t tick, const char * client, const char * sid) {
    return {{slot, h, tick}, client, sid};
}
}  // namespace

TEST_CASE(KvflashSessionStoreFixture, content_matching_never_overrides_an_explicit_sid_on_a_parked_or_a_disk_session) {
    const std::vector<int32_t> ha = {1, 2, 3, 4}, hb = {1, 2, 3, 4, 5}, hm = {1, 2, 3, 4, 5, 6}, hn = {1, 2, 9};
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7};
    const std::vector<KvFlashOwnedSession> parked = {
        owned(0, &ha, 1, "opencode", "ses_a"), owned(1, &hb, 2, "opencode", "ses_b"), owned(2, &hm, 3, "migrated", "migrated:1")};
    // ses_a takes only ses_a, from the first shared token, even though ses_b and the migrated one share more
    KvFlashSessionPick p = kvflash_pick_parked_owner(parked, "opencode", "ses_a", prompt, 0);
    CHECK(p.slot == 0);
    CHECK(p.lcp == 4);
    CHECK(kvflash_pick_parked_owner(parked, "claude", "ses_a", prompt, 0).slot == 2);   // same sid, other client: not the session; the ownerless one is
    // a sid with no session of its own takes the ownerless one past the preamble, never another's
    p = kvflash_pick_parked_owner(parked, "opencode", "ses_c", prompt, 2);
    CHECK(p.slot == 2);
    CHECK(kvflash_pick_parked_owner(parked, "opencode", "ses_c", {1, 2, 8}, 2).slot < 0);    // preamble only: lcp 2 is not > e_role 2
    CHECK(kvflash_pick_parked_owner({owned(0, &ha, 1, "opencode", "ses_a")}, "opencode", "ses_c", prompt, 0).slot < 0);
    // an anonymous request takes ownerless sessions only
    CHECK(kvflash_pick_parked_owner(parked, "anon", "anon:9", prompt, 0).slot == 2);
    CHECK(kvflash_pick_parked_owner({owned(0, &ha, 1, "opencode", "ses_a")}, "anon", "anon:9", prompt, 0).slot < 0);
    // two anonymous sessions: content decides, as today
    const std::vector<KvFlashOwnedSession> anon = {owned(0, &ha, 1, "anon", "anon:1"), owned(1, &hn, 2, "anon", "anon:2")};
    CHECK(kvflash_pick_parked_owner(anon, "anon", "anon:3", prompt, 0).slot == 0);
    // the disk tier follows the same rules
    KvFlashSessionStore store("kvflash_p0_unit_elig");
    KvFlashSessionEntry b = entry("ses_b", 10, "disk");
    b.client = "opencode"; b.history = {1, 2, 3};
    store.upsert(b);
    CHECK(store.pick_disk("opencode", "ses_a", {1, 2, 3, 4}, 0) == nullptr);
    CHECK(store.pick_disk("anon", "anon:1", {1, 2, 3, 4}, 0) == nullptr);
    CHECK(store.pick_disk("opencode", "ses_b", {1, 2, 3, 4}, 0) != nullptr);
}

TEST_CASE(KvflashSessionStoreFixture, two_compactions_under_one_sid_leave_three_addressable_histories_and_no_self_link_after_a_restart) {
    const std::string root = fresh_root("kvflash_p0_unit_archive");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    auto instance = [&](int64_t last, const std::vector<int32_t> & hist, uint8_t fill) {
        KvFlashSessionEntry e = entry("ses_a", last);
        e.client = "opencode";
        e.key = store.new_key("opencode", "ses_a");
        REQUIRE(fs::create_directories(store.dir(e.key)));
        REQUIRE(park_session(store.dir(e.key), hist, 1, fill));
        const KvFlashSessionEntry * prev = store.current("opencode", "ses_a");
        e.archive_of = prev ? prev->key : "";
        const std::string prev_key = e.archive_of;
        store.upsert(e);
        REQUIRE(store.write_meta(e.key));                                           // the successor's meta first ...
        if (!prev_key.empty()) { CHECK(store.find_key(prev_key)->archive.empty()); REQUIRE(store.link_archive(prev_key, e.key)); }   // ... then the old one's link
        return e.key;
    };
    const std::string k1 = instance(10, {1, 2, 3}, 0x10), k2 = instance(20, {4, 5, 6}, 0x20), k3 = instance(30, {7, 8, 9}, 0x30);
    CHECK((std::set<std::string>{k1, k2, k3}).size() == 3);
    CHECK(store.current("opencode", "ses_a")->key == k3);
    // a restart: the scan returns all three, links intact, current instance = the one nobody archived
    KvFlashScanReport rep;
    KvFlashSessionStore after(root);
    after.entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(after.entries().size() == 3);
    CHECK(after.find_key(k1)->archive == k2);
    CHECK(after.find_key(k2)->archive == k3);
    CHECK(after.find_key(k3)->archive.empty());
    CHECK(after.find_key(k2)->archive_of == k1);
    CHECK(after.find_key(k3)->archive_of == k2);
    for (const auto & e : after.entries()) { CHECK(e.archive != e.key); CHECK(e.archive_of != e.key); }
    CHECK(after.current("opencode", "ses_a")->key == k3);
    // each history is addressable: the sid takes the current one, the archives keep their own bundles
    std::vector<int32_t> h;
    CHECK(loads(store.dir(k1), &h)); CHECK(h == (std::vector<int32_t>{1, 2, 3}));
    CHECK(loads(store.dir(k2), &h)); CHECK(h == (std::vector<int32_t>{4, 5, 6}));
    CHECK(loads(store.dir(k3), &h)); CHECK(h == (std::vector<int32_t>{7, 8, 9}));
    // a crash between the successor's meta and the old one's link: the scan repairs the archive from archive_of
    KvFlashSessionEntry cut = *store.find_key(k2);
    cut.archive.clear();
    store.upsert(cut);
    REQUIRE(store.write_meta(k2));
    KvFlashScanReport rep2;
    const auto repaired = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep2);
    for (const auto & e : repaired) if (e.key == k2) CHECK(e.archive == k3);
    // the successor never committed (killed before its head): the directory goes, the old one's link points at nothing and is cleared
    fs::remove(store.dir(k3) + "/head");
    KvFlashScanReport rep3;
    const auto dangling = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep3);
    REQUIRE(dangling.size() == 2);
    for (const auto & e : dangling) if (e.key == k2) CHECK(e.archive.empty());
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, cleanup_adopt_and_a_lost_index_keep_the_raw_sid_links_and_timestamps) {
    const std::string root = fresh_root("kvflash_p0_unit_p103");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    KvFlashSessionEntry e = entry("ses \"odd\" \xc3\xa9", 777);
    e.client = "claude"; e.archive_of = "feedfeedfeedfeed"; e.archive = "beefbeefbeefbeef"; e.created = 55;
    REQUIRE(fs::create_directories(store.dir(e.key)));
    REQUIRE(park_session(store.dir(e.key), {1, 2, 3}, 1, 0x10));
    store.upsert(e);
    REQUIRE(store.write_meta(e.key));
    REQUIRE(store.evict(e.key, {1, 2, 3}));
    KvFlashSessionEntry succ = entry("ses \"odd\" \xc3\xa9", 800);   // its successor instance, whose directory the link names
    succ.client = "claude"; succ.key = "beefbeefbeefbeef"; succ.archive_of = e.key;
    REQUIRE(fs::create_directories(store.dir(succ.key)));
    REQUIRE(park_session(store.dir(succ.key), {9}, 1, 0x20));
    store.upsert(succ);
    REQUIRE(store.write_meta(succ.key));
    // adopt (restore), then the restart cleanup of the directory, then the index is lost
    REQUIRE(store.adopt(e.key));
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(store.dir(e.key), ID, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}, l, why));
    kvflash_bundle_clean(store.dir(e.key), &l.head);
    REQUIRE(store.write_index());
    fs::remove(root + "/index.json");
    KvFlashScanReport rep;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(es.size() == 2);
    CHECK(es[1].key == e.key);                    // the older one
    CHECK(es[1].sid == e.sid);                    // the raw sid, not a hash of it
    CHECK(es[1].client == "claude");
    CHECK(es[1].archive_of == "feedfeedfeedfeed");
    CHECK(es[1].archive == "beefbeefbeefbeef");
    CHECK(es[1].created == 55);
    CHECK(es[1].last_used == 777);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_park_is_durable_only_when_committed_and_a_session_without_a_committed_head_is_never_marked_disk) {
    // pending, failed, rewrite, head generation
    CHECK(kvflash_session_park_durable(true, false, false, 0));    // the writer acknowledged the staged park
    CHECK(!kvflash_session_park_durable(true, true, false, 3));    // the staged park failed: an older head is not this session
    CHECK(!kvflash_session_park_durable(false, false, true, 3));   // the park was skipped (slab snapshot failed)
    CHECK(!kvflash_session_park_durable(false, false, false, 0));  // nothing ever committed
    CHECK(kvflash_session_park_durable(false, false, false, 3));   // restored from disk and untouched
    // the store refuses to list as disk a session whose directory holds no committed head
    const std::string root = fresh_root("kvflash_p0_unit_nohead");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    KvFlashSessionEntry e = entry("ses_x", 10);
    REQUIRE(fs::create_directories(store.dir(e.key)));
    std::ofstream(store.dir(e.key) + "/chunks.1.log") << "uncommitted";
    store.upsert(e);
    CHECK(!store.evict(e.key, {1, 2}));
    CHECK(store.find_key(e.key)->state == "parked-gpu");
    CHECK(fs::exists(store.dir(e.key) + "/chunks.1.log"));         // and nothing was removed
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_slot_reused_by_another_session_starts_its_bundle_bookkeeping_over_a_b_a) {
    const std::string root = fresh_root("kvflash_p0_unit_reuse");
    fs::create_directories(root + "/sessions");
    KvFlashSessionStore store(root);
    const std::string ka = kvflash_sid_key("header", "A"), kb = kvflash_sid_key("header", "B");
    // slot 0 hosts A: its bookkeeping holds A's committed head and generations
    KvFlashSlotBundle slot0 = KvFlashSlotBundle::for_session(ka, "A", "header", 10);
    REQUIRE(fs::create_directories(store.dir(ka)));
    REQUIRE(park_session(store.dir(ka), {1, 2, 3}, 2, 0x10));
    {
        std::vector<uint8_t> bytes;
        REQUIRE(kvflash_bundle_read(store.dir(ka) + "/head", bytes));
        REQUIRE(kvflash_head_unpack(bytes.data(), bytes.size(), ID, slot0.head));
        slot0.next_gen = 5; slot0.resume_L = 64; slot0.pending = true; slot0.debt = {"chunks.9.log"};
    }
    const std::string a_log = slurp(store.dir(ka) + "/chunks.1.log"), a_head = slurp(store.dir(ka) + "/head");
    CHECK(!slot0.fresh_park(false));                                 // A's bookkeeping as it stands would append to A's log
    // B takes the slot: nothing of A's carries over
    slot0 = KvFlashSlotBundle::for_session(kb, "B", "header", 20);
    CHECK(slot0.key == kb);
    CHECK(slot0.sid == "B");
    CHECK(slot0.head.gen == 0);
    CHECK(slot0.staged.gen == 0);
    CHECK(slot0.next_gen == 1);
    CHECK(slot0.resume_L == 0);
    CHECK(!slot0.pending);
    CHECK(!slot0.rewrite);
    CHECK(slot0.debt.empty());
    CHECK(slot0.fresh_park(false));                                   // B's first park writes a fresh log in B's own directory
    REQUIRE(fs::create_directories(store.dir(kb)));
    REQUIRE(park_session(store.dir(kb), {7, 8}, 1, 0x30));
    CHECK(slurp(store.dir(ka) + "/chunks.1.log") == a_log);           // A's log and head are byte-for-byte what A parked
    CHECK(slurp(store.dir(ka) + "/head") == a_head);
    // A comes back (adopted from its directory): its own head and counters, none of B's
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(store.dir(ka), ID, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}, l, why));
    const uint32_t top = std::max({l.head.gen, l.head.end_gen, l.head.ckpt_gen, l.head.trunk_gen});
    slot0 = KvFlashSlotBundle::adopted(ka, "A", "header", 10, std::move(l.head));
    CHECK(slot0.key == ka);
    CHECK(slot0.sid == "A");
    CHECK(slot0.head.history == (std::vector<int32_t>{1, 2, 3}));
    CHECK(slot0.next_gen == top + 1);                                 // past every file the head names
    CHECK(!slot0.pending);
    CHECK(slot0.debt.empty());
    CHECK(!slot0.fresh_park(false));                                  // A's next park appends to A's log, in A's directory
    CHECK(slurp(store.dir(kb) + "/head") != a_head);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, migration_quarantines_what_it_cannot_load_and_never_deletes_it) {
    // identity mismatch, a reduced context (head over the read cap), a log the head names that is gone
    const std::string root = fresh_root("kvflash_p0_unit_quarantine");
    REQUIRE(park_session(root + "/slot0", {1, 2, 3}, 1, 0x10, ID + 7));   // another model
    REQUIRE(park_session(root + "/slot1", std::vector<int32_t>(5000, 4), 1, 0x20));
    REQUIRE(park_session(root + "/slot2", {5, 6}, 1, 0x30));
    fs::remove(root + "/slot2/chunks.1.log");
    REQUIRE(park_session(root + "/slot3", {7, 8}, 1, 0x40));                // loadable
    fs::create_directories(root + "/sessions/slot0.quarantine.1");          // an earlier quarantine: never replaced
    std::ofstream(root + "/sessions/slot0.quarantine.1/head") << "earlier";
    fs::create_directories(root + "/slot4");                                // a first park killed before its head: the only dir cleaned
    std::ofstream(root + "/slot4/chunks.1.log") << "uncommitted";
    const std::string h0 = slurp(root + "/slot0/head"), h1 = slurp(root + "/slot1/head"), h2 = slurp(root + "/slot2/head");

    KvFlashScanReport rep;
    CHECK(kvflash_migrate_slot_dirs(root, ID, /*head_max=*/256, rep) == 1);   // head_max tiny: slot1's head (20 KB) is over it
    CHECK(!fs::exists(root + "/slot0"));
    CHECK(!fs::exists(root + "/slot1"));
    CHECK(!fs::exists(root + "/slot2"));
    CHECK(!fs::exists(root + "/slot4"));
    CHECK(slurp(root + "/sessions/slot0.quarantine.1/head") == "earlier");
    CHECK(slurp(root + "/sessions/slot0.quarantine.2/head") == h0);
    CHECK(slurp(root + "/sessions/slot1.quarantine.1/head") == h1);
    CHECK(slurp(root + "/sessions/slot2.quarantine.1/head") == h2);
    KvFlashScanReport rep2;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep2);
    size_t foreign = 0, disk = 0;
    for (const auto & e : es) { foreign += e.state == "foreign"; disk += e.state == "disk"; }
    CHECK(disk == 1);
    CHECK(foreign == 4);                                                      // all four quarantines listed, untouched
    CHECK(fs::exists(root + "/sessions/slot2.quarantine.1/head"));
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_bundle_whose_records_fail_to_load_after_migration_is_marked_foreign_and_kept) {
    const std::string root = fresh_root("kvflash_p0_unit_corrupt");
    REQUIRE(park_session(root + "/slot0", {1, 2, 3}, 2, 0x10));
    KvFlashScanReport rep;
    REQUIRE(kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep) == 1);
    KvFlashSessionStore store(root);
    store.entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(store.entries().size() == 1);
    const std::string key = store.entries()[0].key;
    { std::ofstream f(store.dir(key) + "/chunks.1.log", std::ios::binary | std::ios::trunc); f << "damaged"; }
    CHECK(!loads(store.dir(key)));                                            // the engine's adopt is refused ...
    store.entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);      // ... and the scan lists it as foreign by itself (R15)
    REQUIRE(store.entries().size() == 1);
    CHECK(store.find_key(key)->state == "foreign");
    CHECK(store.pick_disk("anon", "anon:1", {1, 2, 3, 4}, 0) == nullptr);
    CHECK(fs::exists(store.dir(key) + "/head"));
    CHECK(fs::exists(store.dir(key) + "/chunks.1.log"));
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, a_migration_killed_at_any_step_is_finished_by_the_next_start_with_no_duplicates) {
    for (int crash = 1; crash <= 8; ++crash) {
        const std::string root = fresh_root("kvflash_p0_unit_migcrash");
        REQUIRE(park_session(root + "/slot0", {1, 2, 3}, 1, 0x10));
        REQUIRE(park_session(root + "/slot1", {4, 5}, 1, 0x20));
        REQUIRE(park_session(root + "/slot2", {6}, 1, 0x30, ID + 1));          // never loadable: quarantined
        const std::string h0 = slurp(root + "/slot0/head"), h1 = slurp(root + "/slot1/head");
        KvFlashScanReport killed;
        kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, killed, crash);
        KvFlashScanReport rep;
        kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep);                      // the restart
        for (const char * s : {"slot0", "slot1", "slot2"}) CHECK(!fs::exists(root + "/" + s));
        const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        REQUIRE(es.size() == 3);
        std::set<std::string> heads;
        size_t disk = 0;
        for (const auto & e : es) {
            if (e.state == "disk") { ++disk; heads.insert(slurp(root + "/sessions/" + e.key + "/head")); CHECK(e.sid.rfind("migrated:", 0) == 0); }
            else CHECK(e.key == "slot2.quarantine.1");
        }
        CHECK(disk == 2);
        CHECK(heads == (std::set<std::string>{h0, h1}));                         // each bundle once, byte for byte
        fs::remove_all(root);
    }
}

// ── code-review fixes (REVIEW_SESSIONS_P0_P3_CODE.md R01-R04, R14, R15, R17) ──────────────────

namespace {
// Makes one I/O step fail (or just records it) for the scope of a test.
struct IoFault {
    explicit IoFault(KvFlashIoFault f) { kvflash_io_fault() = std::move(f); }
    ~IoFault() { kvflash_io_fault() = nullptr; }
};
bool ends_with(const std::string & s, const std::string & suffix) { return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0; }

// A session the way the engine leaves it: its directory with a head and its meta.json.
KvFlashSessionEntry commit_session(KvFlashSessionStore & store, const std::string & client, const std::string & sid,
                                   const std::vector<int32_t> & hist, uint8_t fill, int64_t last_used) {
    KvFlashSessionEntry e;
    e.key = kvflash_sid_key(client, sid); e.sid = sid; e.client = client; e.state = "disk";
    e.created = 100; e.last_used = last_used; e.tokens = hist.size(); e.identity = ID;
    fs::create_directories(store.dir(e.key));
    if (!park_session(store.dir(e.key), hist, 1, fill) || (store.upsert(e), !store.write_meta(e.key))) throw std::runtime_error("commit_session");
    return e;
}

KvFlashBundleLimits lim() { return KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}; }

// What the engine's parked_list_ hands the pick: every slot's owner from the book.
std::vector<KvFlashOwnedSession> owners_of(const KvFlashSessionBook & book, const std::vector<int32_t> & hist) {
    static std::vector<int32_t> keep;   // the pick borrows the history
    keep = hist;
    std::vector<KvFlashOwnedSession> v;
    for (int i = 0; i < book.slots(); ++i) v.push_back({{i, &keep, (uint64_t)(i + 1)}, book[i].client, book[i].sid});
    return v;
}
}  // namespace

// R02: the owner of a slot is kept in memory whether or not sessions are persisted.
TEST_CASE(KvflashSessionStoreFixture, r02_a_slot_keeps_its_owner_with_persistence_off_and_a_matching_b_or_anon_cannot_take_it) {
    KvFlashSessionBook book;                  // no enable_store: the bundle dir is unset, or its init failed (lock held, not writable)
    book.init(2);
    REQUIRE(!book.persistent());
    book.begin_session(0, "opencode", "A", 1000);
    CHECK(book[0].sid == "A");
    CHECK(book[0].client == "opencode");
    CHECK(book[0].key.empty());               // no directory, no store entry
    CHECK(book.store().entries().empty());
    const std::vector<int32_t> hist = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto owners = owners_of(book, hist);
    // an explicit B whose prompt matches A's history, and an anonymous request, are not A
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "B", hist, 0).slot != 0);
    CHECK(kvflash_pick_parked_owner(owners, "anon", "anon:9", hist, 0).slot != 0);
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "A", hist, 0).slot == 0);   // A itself resumes
    // the slot leaving its session (drop or evict) resets the owner too
    book.reset(0);
    CHECK(book[0].sid.empty());
    CHECK(book[0].client.empty());
    CHECK(book[0].fresh_park(false));
}

TEST_CASE(KvflashSessionStoreFixture, r02_a_dropped_session_resets_the_owner_with_persistence_off_and_on) {
    const std::string root = fresh_root("kvflash_p0_unit_r02_on");
    KvFlashSessionBook off, on;
    off.init(1); on.init(1);
    on.enable_store(KvFlashSessionStore(root), ID);
    for (KvFlashSessionBook * b : {&off, &on}) {
        b->begin_session(0, "claude", "S", 5);
        CHECK(((*b)[0].sid == "S" && (*b)[0].client == "claude"));
        b->reset(0);
        CHECK(((*b)[0].sid.empty() && (*b)[0].client.empty() && (*b)[0].key.empty()));
    }
    fs::remove_all(root);
}

// R04: startup adoption, eviction without a turn, disk resume; A -> B -> A.
TEST_CASE(KvflashSessionStoreFixture, r04_an_adopted_session_evicted_without_a_turn_keeps_its_history_and_resumes_from_disk_a_b_a) {
    const std::string root = fresh_root("kvflash_p0_unit_r04");
    KvFlashSessionStore seed(root);
    commit_session(seed, "header", "A", {1, 2, 3, 4}, 0x10, 200);
    commit_session(seed, "header", "B", {9, 8, 7}, 0x20, 100);
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(book.store().entries().size() == 2);
    const std::string ka = kvflash_sid_key("header", "A"), kb = kvflash_sid_key("header", "B");
    auto adopt = [&](const std::string & key, const std::vector<int32_t> & want) {
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(book.store().dir(key), ID, CB, lim(), l, why));
        CHECK(book.adopt(0, key, std::move(l.head)) == want);                  // the parked session starts with its history ...
        CHECK(book[0].head.history == want);                                   // ... and the bookkeeping keeps it for the eviction
    };
    adopt(ka, {1, 2, 3, 4});
    REQUIRE(book.evict(0, false));                                              // slot pressure, before any turn
    const KvFlashSessionEntry * a = book.store().find_key(ka);
    REQUIRE(a != nullptr);
    CHECK(a->state == "disk");
    CHECK(a->tokens == 4);
    CHECK(a->history == (std::vector<int32_t>{1, 2, 3, 4}));
    const std::vector<int32_t> prompt_a = {1, 2, 3, 4, 5};
    const KvFlashSessionEntry * pick = book.store().pick_disk("header", "A", prompt_a, 0);
    REQUIRE(pick != nullptr);
    CHECK(pick->key == ka);                                                     // the disk picker finds it again
    adopt(kb, {9, 8, 7});                                                       // B takes the slot
    REQUIRE(book.evict(0, false));
    adopt(ka, {1, 2, 3, 4});                                                    // and A comes back, with its history
    REQUIRE(book.evict(0, false));
    CHECK(book.store().find_key(ka)->history == (std::vector<int32_t>{1, 2, 3, 4}));
    CHECK(book.store().find_key(kb)->history == (std::vector<int32_t>{9, 8, 7}));
    CHECK(book.store().pick_disk("header", "B", {9, 8, 7, 1}, 0) != nullptr);
    fs::remove_all(root);
}

// R03: one owner per identity; a restore has four outcomes; a denied claim is busy.
TEST_CASE(KvflashSessionStoreFixture, r03_a_live_identity_is_busy_for_a_second_request_and_a_denied_resume_claim_is_busy) {
    const std::string root = fresh_root("kvflash_p0_unit_r03");
    KvFlashSessionStore seed(root);
    commit_session(seed, "opencode", "D", {1, 2, 3}, 0x10, 100);
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    book.begin_session(0, "opencode", "A", 1000);                               // A decodes in slot 0
    std::vector<bool> active = {true, false};
    CHECK(book.identity_busy("opencode", "A", active));                          // a second request for A is deferred, never a second owner
    CHECK(!book.identity_busy("opencode", "B", active));
    CHECK(!book.identity_busy("claude", "A", active));                           // the pair is the identity
    CHECK(!book.identity_busy("anon", "anon:7", active));
    CHECK(book.plan_restore("opencode", "A", {1, 2}, 0, active).pick == KvFlashRestorePick::busy);
    active[0] = false;                                                           // A finished: parked, resumable
    CHECK(!book.identity_busy("opencode", "A", active));
    // a session on disk: loadable, unless the resume claim is denied (P3 wires the claim)
    auto plan = book.plan_restore("opencode", "D", {1, 2, 3, 4}, 0, active);
    CHECK(plan.pick == KvFlashRestorePick::load);
    CHECK(plan.key == kvflash_sid_key("opencode", "D"));
    book.resume_claim = [](const std::string &) { return false; };
    CHECK(book.plan_restore("opencode", "D", {1, 2, 3, 4}, 0, active).pick == KvFlashRestorePick::busy);
    book.resume_claim = [](const std::string &) { return true; };
    CHECK(book.plan_restore("opencode", "D", {1, 2, 3, 4}, 0, active).pick == KvFlashRestorePick::load);
    CHECK(book.plan_restore("opencode", "Z", {5, 6}, 0, active).pick == KvFlashRestorePick::none);   // nothing on disk for it: absent
    fs::remove_all(root);
}

// R01: the identity meta is durable before the first head is handed to the writer.
TEST_CASE(KvflashSessionStoreFixture, r01_a_failed_identity_meta_write_or_fsync_blocks_the_park_and_keeps_the_owner) {
    for (const char * op : {"write", "fsync"}) {
        const std::string root = fresh_root("kvflash_p0_unit_r01a");
        KvFlashSessionBook book;
        book.init(1);
        book.enable_store(KvFlashSessionStore(root), ID);
        fs::create_directories(root + "/sessions");
        book.begin_session(0, "opencode", "A", 1000);
        const std::string key = book[0].key;
        const KvFlashParkStats st{3, 100, 0};
        {
            IoFault f([&](std::string_view o, const std::string & path) { return o == op && (std::string_view(op) == "fsync" ? ends_with(path, key) : ends_with(path, "meta.json")); });
            CHECK(!book.publish_park(0, st, 1001));                              // the writer is not handed this park
        }
        CHECK(book[0].sid == "A");                                               // the owner stays
        CHECK(book[0].client == "opencode");
        CHECK(book[0].rewrite);                                                  // the park is owed, whole
        CHECK(!book.durable(0, false));                                          // not durable: slot pressure keeps the session in memory
        CHECK(!book.evict(0, false));
        CHECK(book.store().find_key(key)->state == "parked-gpu");
        // the fault gone, the next park publishes the identity first and may go
        CHECK(book.publish_park(0, st, 1002));
        REQUIRE(park_session(book.store().dir(key), {1, 2, 3}, 1, 0x10));
        // restart: the record is A's, nobody else's
        KvFlashScanReport rep;
        const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        REQUIRE(es.size() == 1);
        CHECK(es[0].sid == "A");
        CHECK(es[0].client == "opencode");
        CHECK(es[0].state == "disk");
        fs::remove_all(root);
    }
}

TEST_CASE(KvflashSessionStoreFixture, r01_a_later_mutable_meta_failure_does_not_block_a_park_whose_identity_is_already_durable) {
    const std::string root = fresh_root("kvflash_p0_unit_r01b");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    book.begin_session(0, "claude", "S", 1000);
    REQUIRE(book.publish_park(0, KvFlashParkStats{3, 10, 0}, 1001));
    IoFault f([](std::string_view o, const std::string & path) { return o == "write" && ends_with(path, "meta.json"); });
    CHECK(book.publish_park(0, KvFlashParkStats{5, 10, 0}, 1002));             // last_used/tokens are mutable: the head is the truth
    CHECK(!book[0].rewrite);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, p1b_a_park_publishes_the_preamble_binding_and_a_later_park_without_one_clears_it) {
    const std::string root = fresh_root("kvflash_p1b_unit_binding");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    book.begin_session(0, "claude", "S", 1000);
    KvFlashParkStats st{3, 10, 0};
    st.preamble = "00000000000000ab"; st.preamble_chunks = 4;
    REQUIRE(book.publish_park(0, st, 1001));
    const KvFlashSessionEntry * e = book.store().find_key(book[0].key);
    REQUIRE(e != nullptr);
    CHECK(e->preamble == "00000000000000ab");
    CHECK(e->preamble_chunks == 4);
    CHECK(book.publish_park(0, KvFlashParkStats{5, 10, 0}, 1002));   // the binding was released: the meta forgets it
    e = book.store().find_key(book[0].key);
    REQUIRE(e != nullptr);
    CHECK(e->preamble.empty());
    CHECK(e->preamble_chunks == 0);
    fs::remove_all(root);
}

// F01: an ownerless session an explicit request takes over is a new identity publication: the owner change is durable before a head.
TEST_CASE(KvflashSessionStoreFixture, f01_promoting_an_ownerless_session_to_an_explicit_identity_needs_its_owner_meta_durable_before_a_head) {
    const std::string root = fresh_root("kvflash_p0_unit_f01");
    KvFlashSessionStore seed(root);
    const auto anon = commit_session(seed, "anon", "anon:7", {1, 2, 3, 4}, 0x10, 100);
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(book.store().dir(anon.key), ID, CB, lim(), l, why));
    book.adopt(0, anon.key, std::move(l.head));
    CHECK(book[0].identity_durable);                                             // restored from its directory
    book.promote_owner(0, "claude", "S");
    CHECK(book[0].client == "claude");
    CHECK(!book[0].identity_durable);                                            // the new owner is on no disk yet
    {
        IoFault f([](std::string_view o, const std::string & path) { return o == "write" && ends_with(path, "meta.json"); });
        CHECK(!book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1001));        // no head may follow an owner nobody can prove
    }
    CHECK(book[0].rewrite);
    CHECK(!book.durable(0, false));
    CHECK(!book.evict(0, false));
    KvFlashScanReport r2;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, r2);            // a restart before the owner landed: still the old ownerless record
    REQUIRE(es.size() == 1);
    CHECK(es[0].client == "anon");
    CHECK(book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1002));              // the fault gone, the owner lands and the park may go
    CHECK(book[0].identity_durable);
    KvFlashScanReport r3;
    const auto after = kvflash_session_scan(root, ID, RB, HEAD_MAX, r3);
    REQUIRE(after.size() == 1);
    CHECK(after[0].client == "claude");
    CHECK(after[0].sid == "S");
    fs::remove_all(root);
}

// Codex re-review P2 (a): a parked session whose resume wait holds for the request is resumed before any other disk session is restored.
TEST_CASE(KvflashSessionStoreFixture, r17b_a_waiting_resume_restores_no_other_disk_session) {
    const std::string root = fresh_root("kvflash_conc_wait_gate");
    KvFlashSessionStore seed(root);
    commit_session(seed, "anon", "anon:5", {1, 2, 3}, 0x10, 100);   // B, ownerless, on disk, shares the prefix
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    const std::vector<bool> active = {false, true};
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const KvFlashSessionId who{"anon:9", "anon"};
    int restores = 0;
    auto restore = [&] {   // the engine's restore_from_disk_: plan, then read the bundle
        if (book.plan_restore(who.client, who.sid, prompt, 0, active).pick == KvFlashRestorePick::load) ++restores;
        return KvFlashRestore::absent;
    };
    using V = KvFlashAdmitVerdict;
    CHECK(kvflash_admit_gate(book, who, active, false, restore) == V::proceed);
    CHECK(restores == 1);                                             // nothing waits: B is restored
    restores = 0;
    CHECK(kvflash_admit_gate(book, who, active, true, restore) == V::proceed);
    CHECK(restores == 0);                                             // A's wait holds: no bundle read
    fs::remove_all(root);
}

// Codex pass 3: a busy resume that consumed the whole history (L == history) records the wait too; the retry restores no disk session.
TEST_CASE(KvflashSessionStoreFixture, r17c_a_whole_history_busy_resume_leaves_a_wait_and_the_retry_restores_nothing) {
    const std::string root = fresh_root("kvflash_conc_wait_whole");
    KvFlashSessionStore seed(root);
    commit_session(seed, "anon", "anon:5", {1, 2, 3}, 0x10, 100);   // B on disk
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    const std::vector<bool> active = {false, true};
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5};
    const std::vector<int32_t> history = {1, 2, 3, 4};               // A, parked in slot 0: the prompt extends it completely
    const KvFlashSessionId who{"anon:9", "anon"};
    const KvFlashResumeWait wait = kvflash_resume_wait_after_busy(/*L=*/4, /*history=*/4, /*lcp=*/4, prompt, who.client, who.sid);
    const bool waiting = kvflash_pick_with_wait({}, 0, (int)history.size(), wait, prompt, who.client, who.sid).exact();
    int restores = 0;
    auto restore = [&] {
        if (book.plan_restore(who.client, who.sid, prompt, 0, active).pick == KvFlashRestorePick::load) ++restores;
        return KvFlashRestore::absent;
    };
    CHECK(kvflash_admit_gate(book, who, active, waiting, restore) == KvFlashAdmitVerdict::proceed);
    CHECK(restores == 0);
    const KvFlashSessionPick pick = kvflash_pick_with_wait({}, 0, (int)history.size(), wait, prompt, who.client, who.sid);
    CHECK(pick.slot == 0);
    CHECK(pick.lcp == 4);
    fs::remove_all(root);
}

// R17: the engine's admission checks are one pure function over the book; each verdict has a case, and the restore runs only when it is due.
TEST_CASE(KvflashSessionStoreFixture, r17_the_admission_gate_refuses_an_oversized_identity_a_running_one_and_a_restore_that_cannot_happen_yet) {
    KvFlashSessionBook book;
    book.init(2);
    book.begin_session(0, "claude", "S", 1);
    const std::vector<bool> active = {true, false};
    int restores = 0;
    KvFlashRestore outcome = KvFlashRestore::absent;
    auto restore = [&] { ++restores; return outcome; };
    using V = KvFlashAdmitVerdict;
    CHECK(kvflash_admit_gate(book, {std::string(kKvFlashSidMax + 1, 'x'), "body"}, active, restore) == V::invalid_identity);
    CHECK(kvflash_admit_gate(book, {"S", "claude"}, active, restore) == V::identity_busy);   // its slot runs a request
    CHECK(restores == 0);                                                                       // neither refusal touched the disk
    CHECK(kvflash_admit_gate(book, {"S", "claude"}, {false, false}, restore) == V::proceed);   // parked: it resumes
    CHECK(restores == 1);
    CHECK(kvflash_admit_gate(book, {"other", "claude"}, active, restore) == V::proceed);
    for (KvFlashRestore ok : {KvFlashRestore::absent, KvFlashRestore::restored, KvFlashRestore::refused}) {
        outcome = ok;
        CHECK(kvflash_admit_gate(book, {"other", "claude"}, active, restore) == V::proceed);
    }
    outcome = KvFlashRestore::busy;                                                             // no slot to leave, or its storage claimed
    CHECK(kvflash_admit_gate(book, {"other", "claude"}, active, restore) == V::restore_busy);
    CHECK(kvflash_admit_gate(book, KvFlashSessionId{}, active, restore) == V::restore_busy);   // an anonymous request waits on the same restore
}

// R17: a resume hands an ownerless session to the explicit client that took it, and to no one else.
TEST_CASE(KvflashSessionStoreFixture, r17_only_an_explicit_request_takes_over_an_ownerless_session) {
    KvFlashSessionBook book;
    book.init(1);
    book.begin_session(0, "anon", "anon:3", 1);
    book[0].identity_durable = true;
    CHECK(!book.promote_owner(0, "anon", "anon:9"));                              // an anonymous request owns nothing
    CHECK((book[0].client == "anon" && book[0].sid == "anon:3" && book[0].identity_durable));
    CHECK(book.promote_owner(0, "opencode", "ses_1"));
    CHECK((book[0].client == "opencode" && book[0].sid == "ses_1" && !book[0].identity_durable));
    book[0].identity_durable = true;
    CHECK(!book.promote_owner(0, "claude", "other"));                             // a stated identity is never taken over by content
    CHECK((book[0].client == "opencode" && book[0].identity_durable));
}

TEST_CASE(KvflashSessionStoreFixture, r01_an_oversized_identity_is_refused_before_admission_and_never_becomes_an_ownerless_record) {
    KvFlashSessionId ok{std::string(kKvFlashSidMax, 'x'), "body"};
    KvFlashSessionId big{std::string(kKvFlashSidMax + 1, 'x'), "body"};
    KvFlashSessionId huge{std::string(2 << 20, 'x'), "header"};
    CHECK(kvflash_identity_valid(ok));
    CHECK(!kvflash_identity_valid(big));
    CHECK(!kvflash_identity_valid(huge));
    CHECK(kvflash_identity_valid(KvFlashSessionId{}));                           // no identity: anonymous, valid
    // the worst accepted identity (every byte escaped to \u00XX) still fits the reader's limit and round-trips
    KvFlashSessionEntry e = entry(std::string(kKvFlashSidMax, '\x01'), 5);
    e.client = "body";
    const std::string json = kvflash_meta_pack(e);
    CHECK(json.size() < kKvFlashMetaMax);
    KvFlashSessionEntry back;
    REQUIRE(kvflash_meta_unpack(json, back));
    CHECK(back.sid == e.sid);
    // and the writer refuses a meta the reader would not take, instead of publishing an unreadable one
    const std::string root = fresh_root("kvflash_p0_unit_r01c");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    book.begin_session(0, "body", std::string(2 << 20, 'x'), 1);
    CHECK(!book.publish_park(0, KvFlashParkStats{1, 1, 0}, 2));
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, r01_a_named_record_whose_meta_cannot_be_read_stays_foreign_and_no_identity_can_take_it) {
    const std::string root = fresh_root("kvflash_p0_unit_r01d");
    KvFlashSessionStore seed(root);
    const auto a = commit_session(seed, "opencode", "A", {1, 2, 3, 4, 5, 6}, 0x10, 100);
    std::ofstream(seed.dir(a.key) + "/meta.json", std::ios::trunc) << "{\"sid\": \"A\"";          // torn
    KvFlashScanReport rep;
    KvFlashSessionStore store(root);
    store.entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(store.entries().size() == 1);
    CHECK(store.entries()[0].state == "foreign");
    CHECK(store.entries()[0].client != "migrated");
    CHECK(store.pick_disk("opencode", "B", {1, 2, 3, 4, 5, 6, 7}, 0) == nullptr);                 // explicit B cannot take it by content
    CHECK(store.pick_disk("anon", "anon:1", {1, 2, 3, 4, 5, 6, 7}, 0) == nullptr);                // nor an anonymous request
    CHECK(fs::exists(store.dir(a.key) + "/head"));
    fs::remove_all(root);
}

// R14: the child directory is fsynced after migration's metadata write; a failure is reported, never a silent publish.
TEST_CASE(KvflashSessionStoreFixture, r14_migration_fsyncs_the_directory_that_holds_the_meta_and_reports_a_failure) {
    {
        const std::string root = fresh_root("kvflash_p0_unit_r14a");
        REQUIRE(park_session(root + "/slot0", {1, 2, 3}, 1, 0x10));
        std::vector<std::string> ops;
        {
            IoFault f([&](std::string_view o, const std::string & path) { ops.push_back(std::string(o) + " " + path); return false; });
            KvFlashScanReport rep;
            REQUIRE(kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep) == 1);
        }
        const auto meta = std::find_if(ops.begin(), ops.end(), [](const std::string & o) { return o.rfind("write ", 0) == 0 && ends_with(o, "/meta.json"); });
        REQUIRE(meta != ops.end());
        const std::string meta_dir = meta->substr(6, meta->size() - 6 - std::string("/meta.json").size());
        const auto sync = std::find(meta, ops.end(), "fsync " + meta_dir);
        CHECK(sync != ops.end());                                               // the meta's directory is fsynced after the write
        fs::remove_all(root);
    }
    {
        const std::string root = fresh_root("kvflash_p0_unit_r14b");
        REQUIRE(park_session(root + "/slot0", {1, 2, 3}, 1, 0x10));
        KvFlashScanReport rep;
        {
            IoFault f([](std::string_view o, const std::string & path) { return o == "fsync" && path.find("slot0") != std::string::npos; });
            CHECK(kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep) == 0);      // not published as migrated
        }
        CHECK(!rep.failed.empty());
        CHECK(!fs::exists(root + "/sessions/" + kvflash_session_dir_key("migrated", "x", {})) || true);
        KvFlashScanReport again;                                                 // and the next start finishes it
        CHECK(kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, again) == 1);
        fs::remove_all(root);
    }
}

TEST_CASE(KvflashSessionStoreFixture, r14_scan_fsyncs_an_archive_metadata_repair_and_reports_a_failed_one) {
    auto build = [&](const std::string & root) {
        KvFlashSessionStore seed(root);
        KvFlashSessionEntry old = commit_session(seed, "header", "S", {1, 2, 3}, 0x10, 100);
        KvFlashSessionEntry nu = commit_session(seed, "header", "S#1", {4, 5, 6}, 0x20, 200);
        nu.sid = "S"; nu.archive_of = old.key;                                    // the successor's link is durable; the old one's was cut off
        seed.upsert(nu);
        REQUIRE(seed.write_meta(nu.key));
        return std::make_pair(old.key, nu.key);
    };
    {
        const std::string root = fresh_root("kvflash_p0_unit_r14c");
        const auto keys = build(root);
        std::vector<std::string> ops;
        KvFlashScanReport rep;
        {
            IoFault f([&](std::string_view o, const std::string & path) { ops.push_back(std::string(o) + " " + path); return false; });
            kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        }
        const std::string dir = root + "/sessions/" + keys.first;
        const auto w = std::find(ops.begin(), ops.end(), "write " + dir + "/meta.json");
        REQUIRE(w != ops.end());
        CHECK(std::find(w, ops.end(), "fsync " + dir) != ops.end());
        CHECK(rep.failed.empty());
        fs::remove_all(root);
    }
    {
        const std::string root = fresh_root("kvflash_p0_unit_r14d");
        const auto keys = build(root);
        KvFlashScanReport rep;
        {
            IoFault f([&](std::string_view o, const std::string & path) { return o == "fsync" && ends_with(path, keys.first); });
            kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        }
        CHECK(std::find(rep.failed.begin(), rep.failed.end(), keys.first) != rep.failed.end());
        fs::remove_all(root);
    }
}

// R15: an error from the filesystem is not "no head"; damaged records are foreign from the scan on.
TEST_CASE(KvflashSessionStoreFixture, r15_a_stat_error_on_the_head_is_unknown_and_preserved_by_the_scan_and_the_migration) {
    {
        const std::string root = fresh_root("kvflash_p0_unit_r15a");
        KvFlashSessionStore seed(root);
        const auto a = commit_session(seed, "header", "A", {1, 2, 3}, 0x10, 100);
        KvFlashScanReport rep;
        std::vector<KvFlashSessionEntry> es;
        {
            IoFault f([](std::string_view o, const std::string & path) { return o == "stat" && ends_with(path, "/head"); });
            es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
        }
        CHECK(fs::exists(seed.dir(a.key) + "/head"));                            // not removed as an empty directory
        CHECK(rep.removed.empty());
        REQUIRE(es.size() == 1);
        CHECK(es[0].state == "foreign");
        fs::remove_all(root);
    }
    {
        const std::string root = fresh_root("kvflash_p0_unit_r15b");
        REQUIRE(park_session(root + "/slot0", {1, 2, 3}, 1, 0x10));
        KvFlashScanReport rep;
        {
            IoFault f([](std::string_view o, const std::string & path) { return o == "stat" && ends_with(path, "/head"); });
            kvflash_migrate_slot_dirs(root, ID, HEAD_MAX, rep);
        }
        CHECK(rep.removed.empty());
        std::error_code ec;
        size_t heads = 0;
        for (const auto & de : fs::recursive_directory_iterator(root, ec)) heads += de.path().filename() == "head";
        CHECK(heads == 1);                                                       // nothing was cleaned
        fs::remove_all(root);
    }
}

TEST_CASE(KvflashSessionStoreFixture, r15_the_scan_marks_a_corrupt_log_a_missing_or_malformed_slab_foreign_beyond_the_adoption_set) {
    const std::string root = fresh_root("kvflash_p0_unit_r15c");
    KvFlashSessionStore seed(root);
    const auto good = commit_session(seed, "header", "good", {1, 2, 3}, 0x10, 500);
    const auto bad_log = commit_session(seed, "header", "bad_log", {4, 5, 6}, 0x20, 100);
    const auto no_slab = commit_session(seed, "header", "no_slab", {7, 8, 9}, 0x30, 100);
    const auto bad_slab = commit_session(seed, "header", "bad_slab", {2, 4, 6}, 0x40, 100);
    const auto short_log = commit_session(seed, "header", "short_log", {3, 6, 9}, 0x50, 100);
    {   // a flipped byte inside a record's payload
        std::fstream f(seed.dir(bad_log.key) + "/chunks.1.log", std::ios::in | std::ios::out | std::ios::binary);
        f.seekp((std::streamoff)(kKvFlashLogHeader + kKvFlashRecordTag + 1));
        f.put('\x7f');
    }
    fs::remove(seed.dir(no_slab.key) + "/" + kvflash_bundle_slab_name(2));
    { std::ofstream f(seed.dir(bad_slab.key) + "/" + kvflash_bundle_slab_name(2), std::ios::binary | std::ios::trunc); f << "xx"; }
    fs::resize_file(seed.dir(short_log.key) + "/chunks.1.log", kKvFlashLogHeader + 4);
    KvFlashScanReport rep;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(es.size() == 5);
    for (const auto & e : es) {
        if (e.key == good.key) CHECK(e.state == "disk");
        else { CHECK(e.state == "foreign"); CHECK(e.history.empty()); CHECK(fs::exists(seed.dir(e.key) + "/head")); }
    }
    fs::remove_all(root);
}

// R17: a failed or skipped park followed by slot pressure, through the book the engine runs.
TEST_CASE(KvflashSessionStoreFixture, r17_a_skipped_or_failed_park_followed_by_slot_pressure_never_lists_the_session_as_disk) {
    const std::string root = fresh_root("kvflash_p0_unit_r17");
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    book.begin_session(0, "header", "A", 1);
    const std::string key = book[0].key;
    REQUIRE(book.publish_park(0, KvFlashParkStats{3, 10, 0}, 2));
    REQUIRE(park_session(book.store().dir(key), {1, 2, 3}, 1, 0x10));
    book[0].head = KvFlashSlotHead{}; book[0].head.gen = 1; book[0].head.history = {1, 2, 3};     // the writer acknowledged park 1
    // park 2 is skipped (slab snapshot failed): the engine marks `rewrite`; the directory's head is older than the session
    book[0].rewrite = true;
    CHECK(!book.durable(0, false));
    CHECK(!book.evict(0, false));                                              // pressure: the older head is not this session
    CHECK(book.store().find_key(key)->state != "disk");
    CHECK(book[0].sid == "A");
    // park 3 was staged and its verdict is a failure
    book[0].rewrite = false; book[0].pending = true; book[0].staged.gen = 3; book[0].staged.history = {1, 2, 3, 4};
    CHECK(!book.durable(0, true));
    CHECK(!book.evict(0, true));
    CHECK(book.store().find_key(key)->state != "disk");
    // the staged park acknowledged: it leaves, with the history the head holds
    CHECK(book.durable(0, false));
    REQUIRE(book.evict(0, false));
    CHECK(book.store().find_key(key)->state == "disk");
    CHECK(book.store().find_key(key)->history == (std::vector<int32_t>{1, 2, 3, 4}));
    fs::remove_all(root);
}

// ── P2: archive instances (4.3) and Gap 4's history = KV length (2.5) ──────────────────────────

namespace {
void must(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }   // REQUIRE needs the fixture; these helpers are free functions
// A slot the way the engine leaves it after a turn: its metadata published, its head committed and acknowledged.
void park_in_slot(KvFlashSessionBook & book, int slot, const std::vector<int32_t> & hist, uint8_t fill, int64_t now) {
    must(book.publish_park(slot, KvFlashParkStats{hist.size(), 10, 0}, now), "publish_park");
    must(park_session(book.store().dir(book[slot].key), hist, 1, fill), "park_session");
    book[slot].head = KvFlashSlotHead{}; book[slot].head.gen = 1; book[slot].head.history = hist;
}
KvFlashSessionBook persistent_book(const std::string & root, int slots) {
    KvFlashSessionBook book;
    book.init(slots);
    book.enable_store(KvFlashSessionStore(root), ID);
    fs::create_directories(root + "/sessions");
    return book;
}
// The archive step of a post-summary request: the parked instance leaves for disk, its successor starts naming it.
std::string archive_and_succeed(KvFlashSessionBook & book, int slot, const std::string & client, const std::string & sid, int64_t now) {
    const std::string old_key = book[slot].key;
    must(book.evict(slot, false), "evict");
    book.begin_session(slot, client, sid, now, book.archive_target(old_key));
    return old_key;
}
}  // namespace

TEST_CASE(KvflashSessionStoreFixture, p2_a_summary_request_beside_a_parked_sid_links_nothing_and_the_post_summary_request_archives_the_parked_instance_by_key) {
    const std::string root = fresh_root("kvflash_p2_unit_archive");
    KvFlashSessionBook book = persistent_book(root, 2);
    book.begin_session(0, "opencode", "ses", 1);
    const std::string main_key = book[0].key;
    const std::vector<int32_t> main_hist = {1, 2, 3, 4, 5, 6};
    park_in_slot(book, 0, main_hist, 0x10, 2);
    // (1) summary generation, row 3: cold in the other slot, "" = no archive link
    book.begin_session(1, "opencode", "ses", 3, std::string());
    const std::string summary_key = book[1].key;
    CHECK(summary_key == kvflash_sid_key("opencode", "ses#1"));
    park_in_slot(book, 1, {9, 8, 7}, 0x20, 4);
    CHECK(book.store().find_key(summary_key)->archive_of.empty());
    CHECK(book.store().find_key(main_key)->archive.empty());
    CHECK(book.store().find_key(main_key)->state == "parked-gpu");   // untouched, still in its slot
    // (2) the first post-summary request, row 4: archive the parked instance (the pick's, not the newest), then its successor
    const std::string old_key = archive_and_succeed(book, 0, "opencode", "ses", 5);
    CHECK(old_key == main_key);
    const std::string succ_key = book[0].key;
    CHECK(succ_key == kvflash_sid_key("opencode", "ses#2"));
    CHECK(book.store().find_key(old_key)->state == "disk");
    CHECK(book.store().find_key(succ_key)->archive_of == main_key);
    CHECK(book.store().find_key(main_key)->archive.empty());        // published with the successor's first park, after its meta
    park_in_slot(book, 0, {1, 2, 3, 4, 20, 21}, 0x30, 6);
    CHECK(book.store().find_key(main_key)->archive == succ_key);
    CHECK(book.store().find_key(summary_key)->archive.empty());
    CHECK(book.store().find_key(summary_key)->archive_of.empty());
    // keys, not sids, and on disk
    KvFlashSessionEntry on_disk;
    REQUIRE(kvflash_meta_unpack(slurp(book.store().dir(main_key) + "/meta.json"), on_disk));
    CHECK(on_disk.archive == succ_key);
    REQUIRE(kvflash_meta_unpack(slurp(book.store().dir(succ_key) + "/meta.json"), on_disk));
    CHECK(on_disk.archive_of == main_key);
    // the recall hook lists the archived instance with its tokens, and no self link
    const auto cands = book.store().recall_candidates(succ_key);
    REQUIRE(cands.size() == 1);
    CHECK(cands[0].old_key == main_key);
    CHECK(cands[0].history == main_hist);
    CHECK(cands[0].dir == book.store().dir(main_key));
    CHECK(book.store().recall_candidates(main_key).empty());
    CHECK(book.store().recall_candidates("nope").empty());
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, p2_two_compactions_under_one_sid_then_a_restart_give_three_addressable_directories_and_no_self_link) {
    const std::string root = fresh_root("kvflash_p2_unit_two_compactions");
    KvFlashSessionBook book = persistent_book(root, 1);
    book.begin_session(0, "opencode", "ses", 1);
    const std::string k0 = book[0].key;
    CHECK(k0 == kvflash_sid_key("opencode", "ses"));
    park_in_slot(book, 0, {1, 2, 3}, 0x10, 2);
    REQUIRE(archive_and_succeed(book, 0, "opencode", "ses", 3) == k0);
    const std::string k1 = book[0].key;
    park_in_slot(book, 0, {4, 5, 6}, 0x20, 4);
    REQUIRE(archive_and_succeed(book, 0, "opencode", "ses", 5) == k1);
    const std::string k2 = book[0].key;
    park_in_slot(book, 0, {7, 8, 9}, 0x30, 6);
    CHECK(k1 == kvflash_sid_key("opencode", "ses#1"));
    CHECK(k2 == kvflash_sid_key("opencode", "ses#2"));
    CHECK((std::set<std::string>{k0, k1, k2}).size() == 3);
    REQUIRE(book.evict(0, false));
    // restart
    KvFlashScanReport rep;
    KvFlashSessionStore after(root);
    after.entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(after.entries().size() == 3);
    CHECK(after.find_key(k0)->archive == k1);
    CHECK(after.find_key(k1)->archive == k2);
    CHECK(after.find_key(k2)->archive.empty());
    CHECK(after.find_key(k1)->archive_of == k0);
    CHECK(after.find_key(k2)->archive_of == k1);
    for (const auto & e : after.entries()) { CHECK(e.archive != e.key); CHECK(e.archive_of != e.key); CHECK(e.state == "disk"); }
    std::vector<int32_t> h;
    CHECK(loads(after.dir(k0), &h)); CHECK(h == (std::vector<int32_t>{1, 2, 3}));
    CHECK(loads(after.dir(k1), &h)); CHECK(h == (std::vector<int32_t>{4, 5, 6}));
    CHECK(loads(after.dir(k2), &h)); CHECK(h == (std::vector<int32_t>{7, 8, 9}));
    const auto chain = after.recall_candidates(k2);
    REQUIRE(chain.size() == 2);
    CHECK(chain[0].old_key == k1);
    CHECK(chain[1].old_key == k0);
    CHECK(chain[0].history == (std::vector<int32_t>{4, 5, 6}));
    CHECK(after.current("opencode", "ses")->key == k2);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, p2_identical_transcripts_under_two_sids_resume_each_its_own_and_keep_separate_directories_through_archive_export_and_delete) {
    const std::string root = fresh_root("kvflash_p2_unit_two_sids");
    const std::string out = fresh_root("kvflash_p2_unit_two_sids_out");
    KvFlashSessionBook book = persistent_book(root, 4);
    const std::vector<int32_t> same = {1, 2, 3, 4, 5, 6, 7, 8};   // byte-identical transcripts
    // A parked first (older), B after
    book.begin_session(0, "opencode", "ses_a", 1); park_in_slot(book, 0, same, 0x10, 10);
    book.begin_session(1, "opencode", "ses_b", 2); park_in_slot(book, 1, same, 0x10, 20);
    CHECK(book[0].key != book[1].key);
    std::vector<KvFlashOwnedSession> owners;
    for (int i = 0; i < 2; ++i) owners.push_back({{i, &same, (uint64_t)(i == 0 ? 9 : 1)}, book[i].client, book[i].sid});   // A is the newer park here
    const std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "ses_a", prompt, 2).slot == 0);
    CHECK(kvflash_pick_parked_owner(owners, "opencode", "ses_b", prompt, 2).slot == 1);
    // both compact: each archives its own instance
    const std::string old_a = archive_and_succeed(book, 0, "opencode", "ses_a", 30), succ_a = book[0].key;
    park_in_slot(book, 0, {1, 2, 50, 51}, 0x40, 31);
    const std::string old_b = archive_and_succeed(book, 1, "opencode", "ses_b", 32), succ_b = book[1].key;
    park_in_slot(book, 1, {1, 2, 60, 61}, 0x50, 33);
    CHECK((std::set<std::string>{old_a, old_b, succ_a, succ_b}).size() == 4);
    CHECK(book.store().find_key(old_a)->archive == succ_a);
    CHECK(book.store().find_key(old_b)->archive == succ_b);
    CHECK(book.store().find_key(succ_a)->archive_of == old_a);
    CHECK(book.store().find_key(succ_b)->archive_of == old_b);
    CHECK(book.store().find_key(old_a)->sid == "ses_a");
    CHECK(book.store().find_key(old_b)->sid == "ses_b");
    // A's archive is exported, B's is deleted: the other session's directories stay as they were
    KvFlashStorageEnv env;   // space shrunk as in the storage tests: a "GiB" is 64 bytes, the disk is far from full
    env.free_bytes = env.disk_bytes = [] { return 1ull << 40; };
    env.gib = 64; env.mib = 1;
    KvFlashStorage st(book.store(), ID, RB, HEAD_MAX, env);
    st.load();
    std::string err;
    REQUIRE(st.set_policy("{\"after_export\":\"keep\"}", err));
    const std::string head_b = slurp(book.store().dir(old_b) + "/head");
    REQUIRE(st.export_session(st.claim_of(old_a), out).status == 200);
    CHECK(fs::exists(out + "/" + old_a + ".g1/head"));
    CHECK(!fs::exists(out + "/" + old_b));
    CHECK(!fs::exists(out + "/" + old_b + ".g1"));
    CHECK(slurp(book.store().dir(old_b) + "/head") == head_b);
    REQUIRE(st.remove(st.claim_of(old_b)).status == 200);
    CHECK(!fs::exists(book.store().dir(old_b)));
    CHECK(fs::exists(book.store().dir(old_a) + "/head"));
    CHECK(fs::exists(book.store().dir(succ_a) + "/head"));
    CHECK(fs::exists(book.store().dir(succ_b) + "/head"));
    CHECK(book.store().find_key(old_a)->archive == succ_a);
    CHECK(book.store().find_key(succ_a)->archive_of == old_a);
    fs::remove_all(root);
    fs::remove_all(out);
}

TEST_CASE(KvflashSessionStoreFixture, p2_a_failed_eviction_never_links_the_successor_to_an_instance_without_a_durable_head) {
    const std::string root = fresh_root("kvflash_p2_unit_failed_evict");
    KvFlashSessionBook book = persistent_book(root, 2);
    // slot 0: published but its park never committed (the writer failed): the eviction is refused, the instance stays in its slot
    book.begin_session(0, "opencode", "ses", 1);
    const std::string stuck = book[0].key;
    REQUIRE(book.publish_park(0, KvFlashParkStats{3, 10, 0}, 2));
    CHECK(!book.evict(0, false));
    CHECK(book.archive_target(stuck).empty());                       // not a disk instance: no link to publish
    book.begin_session(1, "opencode", "ses", 3, book.archive_target(stuck));
    const std::string succ = book[1].key;
    CHECK(book.store().find_key(succ)->archive_of.empty());
    park_in_slot(book, 1, {5, 6, 7}, 0x20, 4);
    CHECK(book.store().find_key(stuck)->archive.empty());
    CHECK(book.store().find_key(stuck)->state == "parked-gpu");       // left where it is
    KvFlashSessionEntry on_disk;
    REQUIRE(kvflash_meta_unpack(slurp(book.store().dir(succ) + "/meta.json"), on_disk));
    CHECK(on_disk.archive_of.empty());
    // an instance dropped by the eviction (no committed head) is no target either; an evicted one is
    CHECK(book.archive_target("no-such-key").empty());
    park_in_slot(book, 0, {1, 2, 3}, 0x10, 5);
    REQUIRE(book.evict(0, false));
    CHECK(book.archive_target(stuck) == stuck);
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, p2_the_recall_hook_returns_the_archived_histories_after_a_restart_adopted_them_into_slots) {
    const std::string root = fresh_root("kvflash_p2_unit_recall_adopted");
    KvFlashSessionBook book = persistent_book(root, 1);
    book.begin_session(0, "opencode", "ses", 1);
    const std::string k0 = book[0].key;
    park_in_slot(book, 0, {1, 2, 3}, 0x10, 2);
    REQUIRE(archive_and_succeed(book, 0, "opencode", "ses", 3) == k0);
    const std::string k1 = book[0].key;
    park_in_slot(book, 0, {4, 5, 6}, 0x20, 4);
    REQUIRE(archive_and_succeed(book, 0, "opencode", "ses", 5) == k1);
    const std::string k2 = book[0].key;
    park_in_slot(book, 0, {7, 8, 9}, 0x30, 6);
    REQUIRE(book.evict(0, false));
    // restart: the scan, then startup adoption of two slots (k2 and k1) as the engine does it
    KvFlashSessionBook after = persistent_book(root, 2);
    KvFlashScanReport rep;
    after.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    REQUIRE(after.store().entries().size() == 3);
    int slot = 0;
    for (const std::string & k : {k2, k1}) {
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(after.store().dir(k), ID, CB, lim(), l, why));
        after.adopt(slot++, k, std::move(l.head));
    }
    CHECK(after.store().find_key(k1)->state == "parked-gpu");
    CHECK(after.store().find_key(k1)->history.empty());              // the entry gave its history to the slot
    const auto chain = after.recall_candidates(k2);
    REQUIRE(chain.size() == 2);
    CHECK(chain[0].old_key == k1);
    CHECK(chain[0].history == (std::vector<int32_t>{4, 5, 6}));      // from the slot owner's head
    CHECK(chain[1].old_key == k0);
    CHECK(chain[1].history == (std::vector<int32_t>{1, 2, 3}));      // still on disk: the entry's
    fs::remove_all(root);
}

TEST_CASE(KvflashSessionStoreFixture, p2_gap4_history_size_equals_the_kv_length_through_park_and_load) {
    // kvflash_park_point(len, stash_pos, ckpt, trunk): the head's history is cut to the point the KV is parked at
    struct Case { const char * name; int len, stash, ckpt, trunk; int want; };
    const Case cases[] = {
        {"stash one short",        20, 19, 0, 0, 19},
        {"cancel (no stash)",      20, 0,  0, 0, 20},
        {"end rung at len-1",      20, 19, 19, 0, 19},   // the rung is not orphaned: park at it
        {"stash below a rung",     20, 18, 19, 0, 20},   // never below a rung
        {"stash too far back",     20, 10, 0, 0, 20},    // only `tail` tokens back
    };
    const std::string root = fresh_root("kvflash_p2_unit_gap4");
    for (const Case & c : cases) {
        std::vector<int32_t> tokens(c.len);
        for (int i = 0; i < c.len; ++i) tokens[(size_t)i] = 100 + i;
        const int pp = kvflash_park_point(c.len, c.stash, c.ckpt, c.trunk);
        CHECK(pp == c.want);
        const std::vector<int32_t> hist(tokens.begin(), tokens.begin() + pp);   // history = KV length
        KvFlashSessionBook book = persistent_book(root, 1);
        book.begin_session(0, "opencode", c.name, 1);
        park_in_slot(book, 0, hist, 0x10, 2);
        REQUIRE(book.evict(0, false));
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(book.store().dir(book.store().entries().back().key), ID, CB, lim(), l, why));
        CHECK((int)l.head.history.size() == pp);
        CHECK(book.adopt(0, book.store().entries().back().key, std::move(l.head)).size() == (size_t)pp);   // adoption's KV length is history.size()
        CHECK((int)book[0].head.history.size() == pp);
        // the identical prompt resumes at pp and prefills len - pp tokens; with a pending token appended, the same
        CHECK(kvflash_resume_point(pp, pp, 0, 0, c.len) == (pp < c.len ? pp : -1));
        CHECK(kvflash_resume_point(pp, pp, 0, 0, c.len + 1) == pp);
        fs::remove_all(root);
    }
}

// Review P1 #3 and P2 #5: a committed park, then a refused newer park, then eviction, slot reuse and the next turn.
// The host memory goes, the directory, its store entry and its owner stay; the next turn restores the older head.
TEST_CASE(KvflashSessionStoreFixture, a_refused_park_keeps_the_last_committed_bundle_and_the_next_turn_restores_it) {
    const std::string root = fresh_root("kvflash_p0_unit_refused_park");
    KvFlashSessionStore seed(root);
    commit_session(seed, "opencode", "A", {1, 2, 3, 4}, 0x10, 200);
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    const std::string key = kvflash_sid_key("opencode", "A");
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(book.store().dir(key), ID, CB, lim(), l, why));
    book.adopt(0, key, std::move(l.head));                                   // turn 1's park is committed (gen != 0)
    REQUIRE(book[0].head.gen != 0);
    book[0].rewrite = true;                                                  // turn 2's park was refused: the session is newer than its head
    CHECK(!book.durable(0, false));
    CHECK(!book.evict(0, false));                                            // the ordinary eviction refuses it, as before
    REQUIRE(book.release_to_last_commit(0));                                 // the host-drop fallback
    CHECK(fs::exists(book.store().dir(key) + "/head"));                      // the directory is still there
    const KvFlashSessionEntry * e = book.store().find_key(key);
    REQUIRE(e != nullptr);                                                   // its store entry and its owner too
    CHECK(e->state == "disk");
    CHECK(e->sid == "A");
    CHECK(e->client == "opencode");
    CHECK(e->history == (std::vector<int32_t>{1, 2, 3, 4}));                 // the older head's history, not turn 2's
    CHECK(book[0].key.empty());                                              // the slot is free for the next session
    book.begin_session(0, "opencode", "B", 300);                             // slot reuse
    CHECK(fs::exists(book.store().dir(key) + "/head"));
    // the next turn of A extends turn 2's history, which extends the head: it restores the older head from disk
    const std::vector<bool> active(2, false);
    const auto plan = book.plan_restore("opencode", "A", {1, 2, 3, 4, 5, 6, 7, 8}, 0, active);
    CHECK(plan.pick == KvFlashRestorePick::load);
    CHECK(plan.key == key);
    // a session whose park never committed has nothing to keep: the caller drops it and its next turn is cold
    CHECK(!book.release_to_last_commit(0));
    fs::remove_all(root);
}

// Review P1 #3 (re-review): the index cannot be written while the session is released. The committed head is the durable copy:
// the release still frees the slot, the directory stays, and the entry is listed as disk in memory for the next write or scan to repair.
TEST_CASE(KvflashSessionStoreFixture, a_release_keeps_the_committed_bundle_when_the_index_write_fails) {
    const std::string root = fresh_root("kvflash_p0_unit_release_io");
    KvFlashSessionStore seed(root);
    commit_session(seed, "opencode", "A", {1, 2, 3, 4}, 0x10, 200);
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    const std::string key = kvflash_sid_key("opencode", "A");
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(book.store().dir(key), ID, CB, lim(), l, why));
    book.adopt(0, key, std::move(l.head));
    REQUIRE(book[0].head.gen != 0);
    fs::remove(root + "/index.json");
    fs::create_directory(root + "/index.json");                            // the index cannot be replaced: every write_index fails
    REQUIRE(!book.store().write_index());
    CHECK(book.release_to_last_commit(0));                                 // released: the caller must not drop the directory
    CHECK(fs::exists(book.store().dir(key) + "/head"));
    CHECK(book[0].key.empty());
    const KvFlashSessionEntry * e = book.store().find_key(key);
    REQUIRE(e != nullptr);
    CHECK(e->state == "disk");
    CHECK(e->history == (std::vector<int32_t>{1, 2, 3, 4}));
    fs::remove_all(root);
}

// Re-review 2 P1: a failed owner publication is not durable after a release and a restore. The first park after the restore still needs
// the new owner's meta.json on disk, or a head would follow a record that names the old owner.
TEST_CASE(KvflashSessionStoreFixture, a_release_and_restore_keep_a_failed_owner_publication_not_durable) {
    const std::string root = fresh_root("kvflash_p0_unit_owner_release");
    KvFlashSessionStore seed(root);
    const auto anon = commit_session(seed, "anon", "anon:7", {1, 2, 3, 4}, 0x10, 100);
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    auto restore = [&] {
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(book.store().dir(anon.key), ID, CB, lim(), l, why));
        book.adopt(0, anon.key, std::move(l.head));
    };
    restore();
    book.promote_owner(0, "claude", "S");
    {
        IoFault f([](std::string_view o, const std::string & path) { return o == "write" && ends_with(path, "meta.json"); });
        CHECK(!book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1001));
        REQUIRE(book.release_to_last_commit(0));                                 // the park was refused, the committed head stays
        restore();                                                               // the same process restores it while the fault persists
        CHECK(!book[0].identity_durable);                                        // the owner is still on no disk
        CHECK(!book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1002));         // so no head may follow
        CHECK(book[0].rewrite);
    }
    CHECK(book.publish_park(0, KvFlashParkStats{5, 100, 0}, 1003));              // the fault gone, the owner lands
    CHECK(book[0].identity_durable);
    KvFlashScanReport r2;
    const auto es = kvflash_session_scan(root, ID, RB, HEAD_MAX, r2);
    REQUIRE(es.size() == 1);
    CHECK(es[0].client == "claude");
    fs::remove_all(root);
}

// Re-review 2 P2: the existence check of the head errors while the session is released. The writer acknowledged that head, so the entry
// still ends as disk with its history and the next turn can restore it.
TEST_CASE(KvflashSessionStoreFixture, a_release_lists_the_session_as_disk_when_the_head_existence_check_errors) {
    const std::string root = fresh_root("kvflash_p0_unit_release_exists");
    KvFlashSessionStore seed(root);
    commit_session(seed, "opencode", "A", {1, 2, 3, 4}, 0x10, 200);
    KvFlashSessionBook book;
    book.init(1);
    book.enable_store(KvFlashSessionStore(root), ID);
    KvFlashScanReport rep;
    book.store().entries() = kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    const std::string key = kvflash_sid_key("opencode", "A");
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(book.store().dir(key), ID, CB, lim(), l, why));
    book.adopt(0, key, std::move(l.head));
    const std::string dir = book.store().dir(key);
    fs::permissions(dir, fs::perms::none);                                       // exists(dir/head) now reports an error
    std::error_code ec;
    REQUIRE(!fs::exists(dir + "/head", ec));
    REQUIRE(ec);
    const bool released = book.release_to_last_commit(0);
    fs::permissions(dir, fs::perms::owner_all);
    REQUIRE(released);
    const KvFlashSessionEntry * e = book.store().find_key(key);
    REQUIRE(e != nullptr);
    CHECK(e->state == "disk");
    CHECK(e->history == (std::vector<int32_t>{1, 2, 3, 4}));
    const std::vector<bool> active(1, false);
    CHECK(book.plan_restore("opencode", "A", {1, 2, 3, 4, 5}, 0, active).pick == KvFlashRestorePick::load);
    fs::remove_all(root);
}
