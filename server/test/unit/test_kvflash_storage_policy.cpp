// Pure unit tests for the KVFlash 2.0 storage policy (kvflash_storage.h), phase P3 of
// trackc/SESSIONS_DESIGN.md: the policy object, the tier state machine, the ask flow (pending
// decisions, grace, explicit delete), generation claims, the compact descriptor and its crash
// points, occupancy from stat with peak reservation, export and import. Real bundle directories
// written through kvflash_bundle_commit; no engine, no GPU. Space is shrunk: in these tests a
// "GiB" is 1 KiB and a "MiB" is 1 byte (KvFlashStorageEnv::gib / mib).
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_storage.h"
#include "../../src/server/sessions_api.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <unistd.h>

using namespace luce::common;
namespace fs = std::filesystem;

#define MUST(x) do { if (!(x)) { std::fprintf(stderr, "fixture step failed: %s (line %d)\n", #x, __LINE__); std::abort(); } } while (0)

namespace {
struct KvflashStoragePolicyFixture {};

constexpr size_t CB = 8;
const size_t RB = kvflash_record_bytes(CB);
constexpr uint64_t ID = 0x5e55;
constexpr uint64_t GIB = 64;   // the policy's GiB in these tests, in bytes
constexpr size_t HEAD_MAX = 1 << 20;

std::vector<uint8_t> hdr44() { std::vector<uint8_t> h(kKvFlashPagerHdr); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t)(0xa0 + i); return h; }

// A whole park of `n` chunks, `live` of them named by the map (the others are dead records in the log).
KvFlashSlotHead park_session(const std::string & dir, const std::vector<int32_t> & hist, int n, uint8_t fill, uint64_t identity = ID, int live = -1) {
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = 1; c.log_gen = 1; c.new_log = true; c.new_dir = true; c.record_bytes = RB;
    KvFlashSlotHead h;
    h.identity = identity; h.gen = 1; h.log_gen = 1; h.end_gen = 2; h.history = hist; h.pager_hdr = hdr44(); h.ledger.assign((size_t)n * 8, 0);
    if (live < 0) live = n;
    for (int i = 0; i < n; ++i) {
        c.records.push_back({(uint32_t)i, std::vector<uint8_t>(CB, (uint8_t)(fill + i))});
        const bool is_live = i >= n - live;
        h.map.push_back(is_live ? kKvFlashLogHeader + (uint64_t)i * RB : 0);
        if (is_live) { h.ledger[(size_t)i * 8] = 1; h.ledger[(size_t)i * 8 + 1] = 1; }
    }
    h.log_bytes = kKvFlashLogHeader + (uint64_t)n * RB;
    c.files = {{kvflash_bundle_slab_name(2), kvflash_slab_pack({{fill, 1}}, {{fill, 2}})}};
    c.head = kvflash_head_pack(h);
    kvflash_bundle_commit(c);
    return h;
}

bool loads(const std::string & dir, std::vector<int32_t> * hist = nullptr, uint64_t identity = ID) {
    KvFlashBundleLoaded l;
    std::string why;
    const KvFlashBundleLimits lim{1 << 20, 1 << 20, 1 << 20};
    if (!kvflash_bundle_load(dir, identity, CB, lim, l, why)) return false;
    if (hist) *hist = l.head.history;
    return true;
}

std::string fresh_root(const std::string & name) {
    fs::remove_all(name);
    fs::create_directories(name);
    return name;
}

std::set<std::string> names(const std::string & dir) {
    std::set<std::string> out;
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(dir, ec)) out.insert(e.path().filename().string());
    return out;
}

std::string slurp(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

uint64_t tree_bytes(const std::string & dir) {
    uint64_t n = 0;
    std::error_code ec;
    for (const auto & e : fs::recursive_directory_iterator(dir, ec)) if (e.is_regular_file(ec)) n += e.file_size(ec);
    return n;
}

// A root with a store, a fake clock and fake free space, and a storage over them.
struct Rig {
    std::string root;
    KvFlashSessionStore store;
    int64_t now = 1000000;
    uint64_t free_b = 1ull << 40, disk_b = 1ull << 40;
    std::set<std::string> unlink_fails;   // file names whose unlink fails
    std::unique_ptr<KvFlashStorage> st;
    KvFlashStorageEnv env;

    explicit Rig(const std::string & name, const char * policy_json = "{}") : root(fresh_root(name)), store(root) {
        fs::create_directories(store.dir(""));
        env.now = [this] { return now; };
        env.free_bytes = [this] { return free_b; };
        env.disk_bytes = [this] { return disk_b; };
        env.gib = GIB; env.mib = 1;
        env.unlink_file = [this](const std::string & p) {
            if (unlink_fails.count(fs::path(p).filename().string())) return false;
            return ::unlink(p.c_str()) == 0 || errno == ENOENT;
        };
        start(policy_json);
    }
    ~Rig() { st.reset(); fs::remove_all(root); }

    void start(const char * policy_json) {
        st = std::make_unique<KvFlashStorage>(store, ID, RB, HEAD_MAX, env);
        st->load();
        std::string err;
        MUST(st->set_policy(policy_json, err));
    }
    // a session in the store and on disk (state "disk" keeps its history in the index, as a scan does)
    std::string add(const std::string & sid, int64_t last_used, int chunks = 40, const std::string & state = "disk", int live = -1, uint64_t identity = ID) {
        const std::string key = store.new_key("header", sid);
        std::vector<int32_t> hist(20);
        for (size_t i = 0; i < hist.size(); ++i) hist[i] = (int32_t)(100 + i + sid.size());
        for (char ch : sid) hist[0] += ch;
        const KvFlashSlotHead h = park_session(store.dir(key), hist, chunks, (uint8_t)sid[0], identity, live);
        KvFlashSessionEntry e;
        e.key = key; e.sid = sid; e.client = "header"; e.state = state; e.created = last_used - 10; e.last_used = last_used;
        e.tokens = hist.size(); e.identity = ID; e.gen = h.gen;
        e.bytes_live = tree_bytes(store.dir(key)); e.history = state == "disk" ? hist : std::vector<int32_t>{};
        store.upsert(e);
        MUST(store.write_meta(key));
        return key;
    }
    KvFlashSessionEntry & ent(const std::string & key) {
        for (auto & e : store.entries()) if (e.key == key) return e;
        MUST(false);
        return store.entries().front();
    }
    uint64_t quota(const char * json) { std::string err; MUST(st->set_policy(json, err)); return st->occupancy().quota; }
    std::vector<KvFlashSessionEntry> rescan() {
        KvFlashScanReport rep;
        return kvflash_session_scan(root, ID, RB, HEAD_MAX, rep);
    }
};

const KvFlashSessionEntry * find(const std::vector<KvFlashSessionEntry> & v, const std::string & key) {
    for (const auto & e : v) if (e.key == key) return &e;
    return nullptr;
}

}  // namespace

// ── the policy object ───────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, the_policy_defaults_are_ask_64_gib_30_days_24_hours_delete_after_export) {
    const KvFlashPolicy p;
    CHECK(p.action == "ask");
    CHECK(p.quota_gib == 64);
    CHECK(p.reserve_gib < 0);   // auto: max(10% of the disk, 20 GiB)
    CHECK(p.max_age_days == 30);
    CHECK(p.grace_hours == 24);
    CHECK(p.export_path.empty());
    CHECK(p.after_export == "delete");
    CHECK(p.log_dead_mib == 512);
    CHECK(p.preamble_mib == 2048);
    const std::string j = kvflash_policy_pack(p);
    for (const char * k : {"\"quota_gib\":64", "\"reserve_gib\":\"auto\"", "\"max_age_days\":30", "\"action\":\"ask\"", "\"grace_hours\":24",
                           "\"export_path\":\"\"", "\"after_export\":\"delete\"", "\"log_dead_mib\":512", "\"preamble_mib\":2048"})
        CHECK(j.find(k) != std::string::npos);
}

TEST_CASE(KvflashStoragePolicyFixture, the_policy_round_trips_and_a_partial_object_changes_only_its_fields) {
    KvFlashPolicy p;
    std::string err;
    REQUIRE(kvflash_policy_apply("{\"quota_gib\": 12, \"action\":\"compact\", \"export_path\":\"/mnt/usb \\\"x\\\"\", \"reserve_gib\": 30}", p, err));
    CHECK(p.quota_gib == 12);
    CHECK(p.action == "compact");
    CHECK(p.reserve_gib == 30);
    CHECK(p.export_path == "/mnt/usb \"x\"");
    CHECK(p.max_age_days == 30);   // untouched
    KvFlashPolicy q;
    REQUIRE(kvflash_policy_apply(kvflash_policy_pack(p), q, err));
    CHECK(kvflash_policy_pack(q) == kvflash_policy_pack(p));
    REQUIRE(kvflash_policy_apply("{\"reserve_gib\":\"auto\"}", q, err));
    CHECK(q.reserve_gib < 0);
    REQUIRE(kvflash_policy_apply("{}", q, err));
}

TEST_CASE(KvflashStoragePolicyFixture, a_bad_policy_value_is_refused_naming_the_field_and_changes_nothing) {
    KvFlashPolicy p;
    const std::string before = kvflash_policy_pack(p);
    std::string err;
    for (const auto & [json, field] : std::vector<std::pair<const char *, const char *>>{
             {"{\"action\":\"shred\"}", "action"}, {"{\"quota_gib\":\"lots\"}", "quota_gib"}, {"{\"quota_gib\":-1}", "quota_gib"},
             {"{\"quota_gib\":1.5}", "quota_gib"}, {"{\"max_age_days\":-3}", "max_age_days"}, {"{\"grace_hours\":\"x\"}", "grace_hours"},
             {"{\"after_export\":\"burn\"}", "after_export"}, {"{\"reserve_gib\":\"some\"}", "reserve_gib"}, {"{\"log_dead_mib\":true}", "log_dead_mib"},
             {"{\"action\":\"export\"}", "export_path"},   // export with nowhere to export to
             {"{\"nonsense\":1}", "nonsense"}, {"[1]", "body"}, {"", "body"},
             {"{\"quota_gib\":9, \"action\":\"shred\"}", "action"}}) {
        err.clear();
        CHECK(!kvflash_policy_apply(json, p, err));
        CHECK(err.find(field) != std::string::npos);
        CHECK(kvflash_policy_pack(p) == before);   // all or nothing: the valid quota of the last case did not land
    }
    REQUIRE(kvflash_policy_apply("{\"action\":\"export\", \"export_path\":\"/x\"}", p, err));   // together they are fine
}

TEST_CASE(KvflashStoragePolicyFixture, the_policy_persists_across_a_restart_and_a_corrupt_file_falls_back_to_the_defaults) {
    Rig r("kvflash_p3_unit_policy_persist", "{\"quota_gib\": 7, \"action\":\"delete-oldest\", \"max_age_days\": 3}");
    CHECK(fs::exists(r.root + "/policy.json"));
    CHECK(names(r.root).count("policy.json.tmp") == 0);
    for (const auto & n : names(r.root)) CHECK(n.find(".tmp") == std::string::npos);
    KvFlashStorage again(r.store, ID, RB, HEAD_MAX, r.env);
    CHECK(again.load());
    CHECK(kvflash_policy_pack(again.policy()) == kvflash_policy_pack(r.st->policy()));
    CHECK(again.policy().quota_gib == 7);
    CHECK(again.policy().action == "delete-oldest");
    std::ofstream(r.root + "/policy.json") << "{\"quota_gib\": 7, \"action\"";
    KvFlashStorage broken(r.store, ID, RB, HEAD_MAX, r.env);
    CHECK(!broken.load());
    CHECK(kvflash_policy_pack(broken.policy()) == kvflash_policy_pack(KvFlashPolicy{}));
}

// ── the state machine ───────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, the_tier_state_machine_moves_live_to_disk_to_compact_to_exported_or_gone) {
    CHECK(kvflash_state_next("parked-gpu", "evict") == "disk");
    CHECK(kvflash_state_next("disk", "resume") == "parked-gpu");
    CHECK(kvflash_state_next("disk", "compact") == "compact");
    CHECK(kvflash_state_next("disk", "export") == "exported");
    CHECK(kvflash_state_next("compact", "export") == "exported");
    CHECK(kvflash_state_next("exported", "import") == "disk");
    CHECK(kvflash_state_next("disk", "delete") == "gone");
    CHECK(kvflash_state_next("compact", "delete") == "gone");
    CHECK(kvflash_state_next("foreign", "delete") == "gone");
    // a live session is never acted on, a compact one has nothing left to compact, a foreign one is only deleted
    for (const char * act : {"compact", "export", "delete"}) CHECK(kvflash_state_next("parked-gpu", act).empty());
    CHECK(kvflash_state_next("compact", "compact").empty());
    CHECK(kvflash_state_next("compact", "resume").empty());   // a compact session comes back through the client's cold admit, not a load
    CHECK(kvflash_state_next("foreign", "compact").empty());
    CHECK(kvflash_state_next("foreign", "export").empty());
    CHECK(kvflash_state_next("disk", "teleport").empty());
}

// ── the ask flow ────────────────────────────────────────────────────────────────────────────

namespace {
// 20 sessions s00..s19 (older first), a live one, and a quota `over_tenths`/10 sessions under the total (one session: the two oldest
// are offered, compaction saving about half of each); the store stays under the 1.1 x hard floor.
std::vector<std::string> scenario(Rig & r, const char * policy_extra = "", std::string * live = nullptr, int over_tenths = 10) {
    std::vector<std::string> keys;
    for (int i = 0; i < 20; ++i) { char sid[8]; std::snprintf(sid, sizeof sid, "s%02d", i); keys.push_back(r.add(sid, 100 + 10 * i, 200)); }
    const std::string l = r.add("s-live", 5, 200, "parked-gpu");   // the oldest of all, but in a slot: never offered
    if (live) *live = l;
    MUST(r.store.write_index());   // the store's own bookkeeping files are part of what it occupies
    const uint64_t one = tree_bytes(r.store.dir(keys[0])), total = r.st->occupancy().total;
    const uint64_t quota = total - one * (uint64_t)over_tenths / 10;
    std::string err;
    MUST(r.st->set_policy(("{\"quota_gib\":" + std::to_string((quota + GIB - 1) / GIB) + policy_extra + "}").c_str(), err));
    MUST(r.st->occupancy().total > r.st->occupancy().quota);
    MUST(r.st->occupancy().total <= r.st->occupancy().quota / 10 * 11);
    return keys;
}
}  // namespace

TEST_CASE(KvflashStoragePolicyFixture, at_quota_the_ask_policy_offers_the_oldest_sessions_and_deletes_nothing) {
    Rig r("kvflash_p3_unit_ask_quota");
    std::string live;
    const auto keys = scenario(r, "", &live);
    const auto work = r.st->tick();
    CHECK(work.empty());
    const auto p = r.st->pending();
    REQUIRE(p.size() == 2);
    CHECK(p[0].key == keys[0]);    // the oldest non-live session first; the live one (older still) is never offered
    CHECK(p[1].key == keys[1]);
    CHECK(p[0].reason == "quota");
    CHECK(p[0].gen == r.ent(keys[0]).gen);
    CHECK(p[0].last_used == 100);
    CHECK(p[0].state == "disk");
    CHECK(p[0].deadline == r.now + 24 * 3600);
    CHECK(p[0].options == (std::vector<std::string>{"compact", "export", "delete"}));
    CHECK(p[0].id != p[1].id);
    r.st->tick();   // a second tick adds nothing, and every directory is still there
    CHECK(r.st->pending().size() == 2);
    for (const auto & k : keys) CHECK(fs::exists(r.store.dir(k) + "/head"));
    CHECK(fs::exists(r.store.dir(live) + "/head"));
    // pending.json carries the same across a restart
    KvFlashStorage again(r.store, ID, RB, HEAD_MAX, r.env);
    again.load();
    REQUIRE(again.pending().size() == 2);
    CHECK(again.pending()[0].id == p[0].id);
    CHECK(again.pending()[0].deadline == p[0].deadline);
    CHECK(again.pending()[1].options == p[1].options);
    // an id is never reused after a restart
    CHECK(again.answer("p3", "delete", "").status == 404);
}

TEST_CASE(KvflashStoragePolicyFixture, past_the_age_limit_the_ask_policy_offers_the_old_session_with_the_age_reason) {
    Rig r("kvflash_p3_unit_ask_age", "{\"quota_gib\": 1000, \"max_age_days\": 30}");
    const std::string old = r.add("sid-old", r.now - 31 * 86400), fresh = r.add("sid-new", r.now - 86400);
    r.st->tick();
    const auto p = r.st->pending();
    REQUIRE(p.size() == 1);
    CHECK(p[0].key == old);
    CHECK(p[0].reason == "age");
    CHECK(fs::exists(r.store.dir(old) + "/head"));
    CHECK(fs::exists(r.store.dir(fresh) + "/head"));
}

TEST_CASE(KvflashStoragePolicyFixture, an_old_session_whose_successor_is_live_is_exempt_from_age_and_quota) {
    Rig r("kvflash_p3_unit_exempt", "{\"quota_gib\": 1, \"max_age_days\": 1}");
    const std::string old = r.add("sid-a", r.now - 90 * 86400);
    const std::string succ = r.add("sid-a#2", r.now - 10, 40, "parked-gpu");
    r.ent(old).archive = succ;
    r.ent(succ).archive_of = old;
    r.st->tick();
    CHECK(r.st->pending().empty());
}

TEST_CASE(KvflashStoragePolicyFixture, with_no_answer_after_the_grace_the_session_is_compacted_never_deleted) {
    Rig r("kvflash_p3_unit_grace");
    const auto keys = scenario(r);
    const std::string a = keys[0];
    const std::vector<int32_t> hist_a = r.ent(a).history;
    r.st->tick();
    REQUIRE(r.st->pending().size() == 2);
    r.now += 24 * 3600 - 1;
    r.st->tick();
    CHECK(r.ent(a).state == "disk");   // not yet
    r.now += 2;
    r.st->tick();
    CHECK(r.ent(a).state == "compact");
    CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json"}));
    CHECK(r.ent(a).tokens == hist_a.size());
    for (const auto & k : keys) CHECK(fs::exists(r.store.dir(k) + "/head"));   // nothing deleted
    for (const auto & p : r.st->pending()) CHECK(p.key != a);
    // the scan after a restart agrees, and the history is the head's
    const auto es = r.rescan();
    REQUIRE(find(es, a) != nullptr);
    CHECK(find(es, a)->state == "compact");
    CHECK(find(es, a)->tokens == hist_a.size());
}

TEST_CASE(KvflashStoragePolicyFixture, a_grace_set_in_seconds_by_the_environment_replaces_the_hours) {
    Rig r("kvflash_p3_unit_grace_s");
    r.env.grace_s = 60;
    r.start("{}");
    const auto keys = scenario(r);
    r.st->tick();
    REQUIRE(!r.st->pending().empty());
    CHECK(r.st->pending()[0].deadline == r.now + 60);
}

TEST_CASE(KvflashStoragePolicyFixture, a_session_is_deleted_only_by_an_explicit_yes_and_only_that_one) {
    Rig r("kvflash_p3_unit_explicit_delete");
    const auto keys = scenario(r);
    for (int i = 0; i < 5; ++i) { r.st->tick(); r.now += 3600; }   // five ticks over quota: no deletion happens by itself
    for (const auto & k : keys) CHECK(fs::exists(r.store.dir(k) + "/head"));
    const auto p = r.st->pending();
    REQUIRE(p.size() == 2);
    CHECK(r.st->answer(p[0].id, "shred", "").status == 400);
    const auto yes = r.st->answer(p[0].id, "delete", "");
    CHECK(yes.status == 200);
    CHECK(!fs::exists(r.store.dir(p[0].key)));
    CHECK(r.store.find_key(p[0].key) == nullptr);
    size_t left = 0;
    for (const auto & k : keys) left += fs::exists(r.store.dir(k) + "/head");
    CHECK(left == keys.size() - 1);
    CHECK(r.st->answer(p[0].id, "delete", "").status == 409);   // answered once
    CHECK(r.st->answer("p9999", "delete", "").status == 404);   // never issued
}

TEST_CASE(KvflashStoragePolicyFixture, answering_compact_compacts_and_answering_export_without_a_path_is_a_400) {
    Rig r("kvflash_p3_unit_answer");
    const auto keys = scenario(r);
    r.st->tick();
    const auto p = r.st->pending();
    REQUIRE(!p.empty());
    CHECK(r.st->answer(p[0].id, "export", "").status == 400);
    CHECK(r.ent(p[0].key).state == "disk");   // a refused answer leaves the decision open
    CHECK(r.st->pending().size() == p.size());
    CHECK(r.st->answer(p[0].id, "compact", "").status == 200);
    CHECK(r.ent(p[0].key).state == "compact");
    CHECK(r.st->pending().size() == p.size() - 1);
}

TEST_CASE(KvflashStoragePolicyFixture, delete_oldest_deletes_exactly_the_oldest_until_under_quota) {
    Rig r("kvflash_p3_unit_delete_oldest");
    std::string live;
    const auto keys = scenario(r, ", \"action\":\"delete-oldest\"", &live, 15);
    r.st->tick();
    CHECK(!fs::exists(r.store.dir(keys[0])));
    CHECK(!fs::exists(r.store.dir(keys[1])));
    CHECK(r.store.find_key(keys[0]) == nullptr);
    for (size_t i = 2; i < keys.size(); ++i) CHECK(fs::exists(r.store.dir(keys[i]) + "/head"));   // exactly the oldest two
    CHECK(fs::exists(r.store.dir(live) + "/head"));   // never a live one
    CHECK(r.st->pending().empty());
}

TEST_CASE(KvflashStoragePolicyFixture, the_compact_policy_compacts_the_oldest_until_under_quota_and_keeps_every_history) {
    Rig r("kvflash_p3_unit_policy_compact");
    const auto keys = scenario(r, ", \"action\":\"compact\"");
    r.st->tick();
    CHECK(r.ent(keys[0]).state == "compact");
    CHECK(r.ent(keys[1]).state == "compact");
    CHECK(r.st->occupancy().total <= r.st->occupancy().quota);
    for (const auto & k : keys) CHECK(fs::exists(r.store.dir(k) + "/head"));
    CHECK(r.ent(keys.back()).state == "disk");
    CHECK(r.st->pending().empty());
}

TEST_CASE(KvflashStoragePolicyFixture, past_the_hard_floor_the_ask_policy_compacts_the_oldest_pending_at_once) {
    Rig r("kvflash_p3_unit_hard_floor", "{\"quota_gib\": 2}");
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200), c = r.add("sid-c", 300), d = r.add("sid-d", 400);
    REQUIRE(r.st->occupancy().total > r.st->occupancy().quota * 11 / 10);
    r.st->tick();
    CHECK(r.ent(a).state == "compact");   // no grace wait
    CHECK(fs::exists(r.store.dir(a) + "/compact.json"));
    for (const std::string & k : {a, b, c, d}) CHECK(fs::exists(r.store.dir(k) + "/head"));
}

TEST_CASE(KvflashStoragePolicyFixture, an_all_compact_store_over_quota_asks_for_export_or_delete_and_never_acts_by_itself) {
    Rig r("kvflash_p3_unit_all_compact", "{\"action\":\"compact\", \"quota_gib\": 1}");
    std::vector<std::string> keys;
    for (int i = 0; i < 3; ++i) { keys.push_back(r.add("sid-" + std::to_string(i), 100 + i)); REQUIRE(r.st->compact(r.st->claim_of(keys.back())).status == 200); }
    REQUIRE(r.st->occupancy().total > r.st->occupancy().quota);   // the compact heads alone are over a quota of 1 GiB (64 bytes)
    r.st->tick();
    const auto p = r.st->pending();
    REQUIRE(!p.empty());
    CHECK(p[0].reason == "quota");
    CHECK(p[0].options == (std::vector<std::string>{"export", "delete"}));
    // the grace passes: still nothing deleted, nothing re-compacted
    r.now += 100 * 3600;
    r.st->tick();
    for (const auto & k : keys) { CHECK(fs::exists(r.store.dir(k) + "/head")); CHECK(r.ent(k).state == "compact"); }
    CHECK(!r.st->pending().empty());
    CHECK(r.st->answer(p[0].id, "compact", "").status == 400);   // not on offer
}

// ── generation claims ───────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, a_click_after_the_session_was_resumed_is_a_409_and_the_resume_supersedes_the_decision) {
    Rig r("kvflash_p3_unit_click_vs_resume");
    scenario(r);
    r.st->tick();
    const auto p = r.st->pending();
    REQUIRE(!p.empty());
    const KvFlashPending first = p[0];
    REQUIRE(r.st->claim_resume(first.key).status == 200);               // the resume claims the key first
    CHECK(r.st->claim_resume(first.key).status == 409);                 // one claim at a time
    CHECK(r.st->answer(first.id, "delete", "").status == 409);          // mid-resume
    CHECK(fs::exists(r.store.dir(first.key) + "/head"));
    r.ent(first.key).state = "parked-gpu";                              // the session is back in a slot
    r.st->release(first.key);
    CHECK(r.st->answer(first.id, "delete", "").status == 409);          // after it: stale
    CHECK(fs::exists(r.store.dir(first.key) + "/head"));
    for (const auto & x : r.st->pending()) CHECK(x.key != first.key);   // the resume dropped the pending decision
}

TEST_CASE(KvflashStoragePolicyFixture, a_deadline_that_passes_during_a_resume_does_not_compact_the_claimed_session) {
    Rig r("kvflash_p3_unit_timeout_vs_resume");
    scenario(r);
    r.st->tick();
    REQUIRE(!r.st->pending().empty());
    const std::string key = r.st->pending()[0].key;
    REQUIRE(r.st->claim_resume(key).status == 200);
    r.now += 25 * 3600;
    r.st->tick();
    CHECK(r.ent(key).state == "disk");
    CHECK(fs::exists(r.store.dir(key) + "/" + kvflash_bundle_log_name(1)));
    CHECK(!fs::exists(r.store.dir(key) + "/compact.json"));
    r.ent(key).state = "parked-gpu";
    r.st->release(key);
    r.st->tick();
    CHECK(r.ent(key).state == "parked-gpu");   // a live session is not compacted afterwards either
    CHECK(!fs::exists(r.store.dir(key) + "/compact.json"));
}

TEST_CASE(KvflashStoragePolicyFixture, a_session_resumed_and_parked_again_before_the_click_makes_the_click_stale) {
    Rig r("kvflash_p3_unit_repark_before_click");
    scenario(r);
    r.st->tick();
    const auto p = r.st->pending();
    REQUIRE(p.size() == 2);
    // resume, park again (new gen, newer last_used), evicted back to disk
    KvFlashSessionEntry & e = r.ent(p[0].key);
    e.gen += 1; e.last_used = r.now; e.state = "disk";
    CHECK(r.st->answer(p[0].id, "delete", "").status == 409);
    CHECK(fs::exists(r.store.dir(p[0].key) + "/head"));
    // same gen and state but a newer last_used is stale too
    r.ent(p[1].key).last_used += 1;
    CHECK(r.st->answer(p[1].id, "compact", "").status == 409);
    CHECK(r.ent(p[1].key).state == "disk");
}

TEST_CASE(KvflashStoragePolicyFixture, a_delete_during_an_export_is_a_409_and_an_export_during_a_delete_never_starts) {
    Rig r("kvflash_p3_unit_delete_vs_export", "{\"after_export\":\"keep\"}");
    const std::string out = fresh_root("kvflash_p3_unit_delete_vs_export_out");
    const std::string a = r.add("sid-a", 100);
    KvFlashExportPlan plan;
    REQUIRE(r.st->export_begin(r.st->claim_of(a), out, plan).status == 200);
    CHECK(r.st->remove(r.st->claim_of(a)).status == 409);            // claimed by the export
    CHECK(r.st->compact(r.st->claim_of(a)).status == 409);
    CHECK(fs::exists(r.store.dir(a) + "/head"));
    const KvFlashResult copied = r.st->export_copy(plan);
    CHECK(copied.status == 200);
    CHECK(r.st->export_commit(plan, copied).status == 200);
    CHECK(fs::exists(out + "/" + a + "/head"));
    // the delete that follows works, and a late export of the deleted key is refused
    const KvFlashClaim late = r.st->claim_of(a);
    CHECK(r.st->remove(late).status == 200);
    KvFlashExportPlan again;
    CHECK(r.st->export_begin(late, out, again).status == 404);
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, a_queued_writer_job_blocks_every_action_on_its_key_and_a_stale_ack_never_resurrects_one) {
    Rig r("kvflash_p3_unit_old_slot_writer");
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200);
    r.st->writer_begin(a);                       // a park of A is queued on a slot that is about to be reused by B
    const KvFlashClaim ca = r.st->claim_of(a);
    CHECK(r.st->remove(ca).status == 409);
    CHECK(r.st->compact(ca).status == 409);
    CHECK(r.st->claim_resume(a).status == 409);
    CHECK(r.st->compact(r.st->claim_of(b)).status == 200);   // B is another key: unaffected
    r.st->writer_ack(a, 0, true);
    r.st->apply_acks();                          // the owner thread reconciles the ack before the key is free
    CHECK(r.st->remove(r.st->claim_of(a)).status == 200);
    r.st->writer_ack(a, 0, true);                // the ack of a job whose session was deleted meanwhile
    r.st->apply_acks();
    r.st->writer_begin(a); r.st->writer_ack(a, 0, true); r.st->apply_acks();
    CHECK(r.store.find_key(a) == nullptr);       // no entry comes back
    CHECK(!fs::exists(r.store.dir(a)));
}

TEST_CASE(KvflashStoragePolicyFixture, with_barriers_one_thread_resuming_and_another_clicking_never_both_win) {
    Rig r("kvflash_p3_unit_threads");
    scenario(r);
    r.st->tick();
    const KvFlashPending first = r.st->pending()[0];
    std::mutex m;
    std::condition_variable cv;
    bool claimed = false, clicked = false;
    std::thread resumer([&] {
        const auto c = r.st->claim_resume(first.key);
        {
            std::lock_guard<std::mutex> lk(m);
            claimed = c.status == 200;
        }
        cv.notify_all();
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return clicked; });   // hold the claim until the click was answered
        lk.unlock();
        r.st->release(first.key);
    });
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return claimed; });
    }
    const int click_status = r.st->answer(first.id, "delete", "").status;
    {
        std::lock_guard<std::mutex> lk(m);
        clicked = true;
    }
    cv.notify_all();
    resumer.join();
    CHECK(click_status == 409);
    CHECK(fs::exists(r.store.dir(first.key) + "/head"));
}

// ── the compact descriptor and its crash points ─────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, the_compact_transition_leaves_head_meta_and_a_descriptor_and_nothing_else) {
    Rig r("kvflash_p3_unit_compact_files");
    const std::string a = r.add("sid-a", 100);
    const std::vector<int32_t> hist = r.ent(a).history;
    const uint64_t before = tree_bytes(r.store.dir(a));
    REQUIRE(r.st->compact(r.st->claim_of(a)).status == 200);
    CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json"}));
    const std::string cj = slurp(r.store.dir(a) + "/compact.json");
    CHECK(cj.find("\"gen\":1") != std::string::npos);
    CHECK(cj.find("\"tokens\":" + std::to_string(hist.size())) != std::string::npos);
    CHECK(cj.find("\"history_sha\":\"" + kvflash_hex16(kvflash_fnv1a(std::string_view((const char *)hist.data(), hist.size() * 4))) + "\"") != std::string::npos);
    CHECK(cj.find(kvflash_bundle_log_name(1)) != std::string::npos);
    CHECK(cj.find(kvflash_bundle_slab_name(2)) != std::string::npos);
    CHECK(r.ent(a).state == "compact");
    CHECK(slurp(r.store.dir(a) + "/meta.json").find("\"compact\"") != std::string::npos);
    CHECK(r.ent(a).bytes_live == tree_bytes(r.store.dir(a)));
    CHECK(tree_bytes(r.store.dir(a)) < before);
    CHECK(!loads(r.store.dir(a)));   // a compact record is not a bundle
    CHECK(r.st->compact(r.st->claim_of(a)).status == 409);   // nothing left to compact
}

TEST_CASE(KvflashStoragePolicyFixture, the_descriptor_is_committed_before_any_unlink_and_recovery_never_loads_a_compact_record) {
    // crash points: 1 before the descriptor, 2 after it, 3.. after each unlink, last after the meta rewrite
    for (int crash = 1; crash <= 6; ++crash) {
        Rig r("kvflash_p3_unit_compact_crash");
        const std::string a = r.add("sid-a", 100), other = r.add("sid-b", 200);
        const std::vector<int32_t> hist = r.ent(a).history;
        r.env.crash_at = crash;
        r.start("{}");
        const auto res = r.st->compact(r.st->claim_of(a));
        const bool done = res.status == 200;   // 2 unlinks (log, slab): crash 5 is after the meta rewrite, 6 never fires
        CHECK((crash >= 6) == done);
        const bool descriptor = fs::exists(r.store.dir(a) + "/compact.json");
        CHECK(descriptor == (crash >= 2));
        // the unlinks follow the descriptor: before it every file is still there
        if (crash == 1) { CHECK(loads(r.store.dir(a))); CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1))); }
        // a restart: the scan, with nothing remembered
        std::vector<KvFlashSessionEntry> es = r.rescan();
        const KvFlashSessionEntry * e = find(es, a);
        REQUIRE(e != nullptr);
        CHECK(e->state == (descriptor ? "compact" : "disk"));
        CHECK(e->tokens == hist.size());
        if (descriptor) {
            CHECK(e->history.empty());   // read from the head alone, never loaded as a bundle
            CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json"}));   // leftovers removed idempotently
            KvFlashSessionStore s2(r.root);
            s2.entries() = es;
            CHECK(s2.pick_disk("header", "sid-a", hist, 0) == nullptr);   // never adopted as a disk session
        } else {
            CHECK(loads(r.store.dir(a)));
        }
        CHECK(find(es, other)->state == "disk");
        CHECK(loads(r.store.dir(other)));
        // the history survives every point
        std::vector<uint8_t> hb;
        KvFlashSlotHead h;
        REQUIRE(kvflash_bundle_read(r.store.dir(a) + "/head", hb, HEAD_MAX));
        REQUIRE(kvflash_head_unpack(hb.data(), hb.size(), ID, h));
        CHECK(h.history == hist);
    }
}

TEST_CASE(KvflashStoragePolicyFixture, a_failed_unlink_keeps_the_commit_and_the_bytes_counted_until_a_later_pass_removes_the_file) {
    Rig r("kvflash_p3_unit_failed_unlink");
    const std::string a = r.add("sid-a", 100);
    const uint64_t log_bytes = fs::file_size(r.store.dir(a) + "/" + kvflash_bundle_log_name(1));
    const uint64_t total_before = r.st->occupancy().total;
    r.unlink_fails = {kvflash_bundle_log_name(1)};
    const auto res = r.st->compact(r.st->claim_of(a));
    CHECK(res.status == 200);                      // the descriptor is the commit
    CHECK(res.error.find(kvflash_bundle_log_name(1)) != std::string::npos);   // and says what is left
    CHECK(r.ent(a).state == "compact");
    CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(r.st->occupancy().total == tree_bytes(r.root));   // stat: the leftover is still occupied
    CHECK(r.st->occupancy().total >= log_bytes);
    r.unlink_fails.clear();
    r.st->tick();                                  // the steward retries what the descriptor lists
    CHECK(!fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json"}));
    CHECK(r.st->occupancy().total == tree_bytes(r.root));
    CHECK(r.st->occupancy().total < total_before);
}

// ── occupancy from stat and peak reservation ────────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, occupancy_is_what_stat_says_not_what_the_meta_files_claim) {
    Rig r("kvflash_p3_unit_occupancy");
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200);
    r.ent(a).bytes_live = 1; r.ent(a).bytes_dead = 0;   // lying meta
    CHECK(r.st->occupancy().total == tree_bytes(r.root));
    fs::create_directories(r.root + "/slot3");           // a foreign directory the migration left in place
    std::ofstream(r.root + "/slot3/head", std::ios::binary) << std::string(5000, 'x');
    CHECK(r.st->occupancy().total == tree_bytes(r.root));
    CHECK(r.st->occupancy().total >= tree_bytes(r.store.dir(a)) + tree_bytes(r.store.dir(b)) + 5000);
    CHECK(r.st->occupancy().free == r.free_b);
    (void)b;
}

TEST_CASE(KvflashStoragePolicyFixture, free_space_just_above_the_reserve_cannot_take_a_large_first_park) {
    Rig r("kvflash_p3_unit_reserve_first_park", "{\"reserve_gib\": 4}");   // 4 GiB reserve in test units
    const uint64_t peak = kvflash_park_peak(/*records*/ 300 * RB, /*slabs*/ 2000, /*head*/ 400, /*old log while a fresh one coexists*/ 0);
    CHECK(peak == 300 * RB + 2000 + 400);
    uint64_t id = 0;
    std::string why;
    r.free_b = 4 * GIB + 10;
    CHECK(!r.st->reserve(peak, id, why));
    CHECK(!why.empty());
    CHECK(r.st->occupancy().reserved_peaks == 0);
    r.free_b = 4 * GIB + peak - 1;
    CHECK(!r.st->reserve(peak, id, why));
    r.free_b = 4 * GIB + peak;
    CHECK(r.st->reserve(peak, id, why));
    CHECK(r.st->occupancy().reserved_peaks == peak);
    r.st->release_reservation(id);
    CHECK(r.st->occupancy().reserved_peaks == 0);
    // a fresh park that rewrites an older log counts both logs
    CHECK(kvflash_park_peak(10 * RB, 100, 50, 7 * RB) == 10 * RB + 100 + 50 + 7 * RB);
}

TEST_CASE(KvflashStoragePolicyFixture, queued_commits_reserve_their_peaks_together_and_a_released_one_makes_room) {
    Rig r("kvflash_p3_unit_reserve_queue", "{\"reserve_gib\": 1}");
    const uint64_t P = 1000;
    r.free_b = GIB + 2 * P + 5;   // room for two of three
    uint64_t i1 = 0, i2 = 0, i3 = 0;
    std::string why;
    CHECK(r.st->reserve(P, i1, why));
    CHECK(r.st->reserve(P, i2, why));
    CHECK(!r.st->reserve(P, i3, why));
    CHECK(r.st->occupancy().reserved_peaks == 2 * P);
    r.st->release_reservation(i1);
    CHECK(r.st->reserve(P, i3, why));
    CHECK(i3 != i2);
    r.st->release_reservation(i2); r.st->release_reservation(i3);
    CHECK(r.st->occupancy().reserved_peaks == 0);
}

TEST_CASE(KvflashStoragePolicyFixture, past_the_hard_floor_new_durable_growth_is_refused) {
    Rig r("kvflash_p3_unit_reserve_hard_floor", "{\"quota_gib\": 2, \"reserve_gib\": 0}");
    r.add("sid-a", 100); r.add("sid-b", 200); r.add("sid-c", 300); r.add("sid-d", 400);
    REQUIRE(r.st->occupancy().total > r.st->occupancy().quota * 11 / 10);
    uint64_t id = 0;
    std::string why;
    CHECK(!r.st->reserve(10, id, why));
    CHECK(why.find("quota") != std::string::npos);
}

TEST_CASE(KvflashStoragePolicyFixture, a_refused_park_never_discards_the_current_owner) {
    Rig r("kvflash_p3_unit_refusal_keeps_owner", "{\"reserve_gib\": 4, \"quota_gib\": 1, \"action\":\"delete-oldest\"}");
    const std::string owner = r.add("sid-owner", 100, 40, "parked-gpu");   // in a slot, its park was skipped
    r.free_b = 4 * GIB;
    uint64_t id = 0;
    std::string why;
    CHECK(!r.st->reserve(5000, id, why));
    // the engine marks the park skipped: not durable, so no eviction, so the slot answers busy
    CHECK(!kvflash_session_park_durable(/*pending*/ false, /*failed*/ false, /*rewrite*/ true, /*head gen*/ 1));
    // and nothing in the store discards a session that is in a slot, whatever the pressure
    for (int i = 0; i < 3; ++i) { r.st->tick(); r.now += 100 * 3600; }
    CHECK(r.ent(owner).state == "parked-gpu");
    CHECK(fs::exists(r.store.dir(owner) + "/head"));
    CHECK(r.st->remove(r.st->claim_of(owner)).status == 409);
}

TEST_CASE(KvflashStoragePolicyFixture, a_log_compaction_without_room_for_the_duplicate_log_becomes_the_compact_tier_when_the_policy_compacts) {
    for (const char * action : {"compact", "ask", "delete-oldest"}) {
        Rig r("kvflash_p3_unit_log_no_room", "{\"reserve_gib\": 1, \"quota_gib\": 1000, \"log_dead_mib\": 1}");
        std::string err;
        REQUIRE(r.st->set_policy((std::string("{\"action\":\"") + action + "\"}").c_str(), err));
        const std::string a = r.add("sid-a", r.now - 1000, 40, "disk", /*live*/ 10);   // 30 dead records
        REQUIRE(r.ent(a).bytes_dead == 0);
        r.st->tick();                          // the scan's bytes_dead are in the entry; set them as the scan would
        r.ent(a).bytes_dead = 30 * RB;
        r.free_b = GIB + 10 * RB + kKvFlashLogHeader - 1;   // one byte short of the live records beside the old log
        r.st->tick();
        const std::string act = action;
        if (act == "compact") {
            CHECK(r.ent(a).state == "compact");
        } else if (act == "ask") {
            CHECK(r.ent(a).state == "disk");
            bool space = false;
            for (const auto & p : r.st->pending()) space |= p.key == a && p.reason == "space";
            CHECK(space);
        } else {
            CHECK(r.ent(a).state == "disk");   // left as it is
            CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
        }
        CHECK(fs::exists(r.store.dir(a) + "/head"));
    }
}

TEST_CASE(KvflashStoragePolicyFixture, a_dead_log_with_room_is_rewritten_by_a_copy_only_commit_that_loads_back) {
    Rig r("kvflash_p3_unit_log_compaction", "{\"reserve_gib\": 1, \"quota_gib\": 1000, \"log_dead_mib\": 1, \"action\":\"compact\"}");
    const std::string a = r.add("sid-a", r.now - 1000, 40, "disk", /*live*/ 10);
    const std::vector<int32_t> hist = r.ent(a).history;
    r.ent(a).bytes_dead = 30 * RB;
    std::vector<uint8_t> before_blob;
    {
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(r.store.dir(a), ID, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}, l, why));
        before_blob = l.pager;
    }
    const uint64_t size_before = fs::file_size(r.store.dir(a) + "/" + kvflash_bundle_log_name(1));
    r.st->tick();
    CHECK(r.ent(a).state == "disk");
    CHECK(!fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));   // the old log is gone
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(r.store.dir(a), ID, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 1 << 20}, l, why));
    CHECK(l.head.history == hist);
    CHECK(l.pager == before_blob);   // the same KV, byte for byte
    CHECK(l.head.gen > 1);
    CHECK(r.ent(a).gen == l.head.gen);
    CHECK(r.ent(a).bytes_dead == 0);
    uint64_t log_now = 0;
    for (const auto & n : names(r.store.dir(a))) if (n.rfind("chunks.", 0) == 0) log_now = fs::file_size(r.store.dir(a) + "/" + n);
    CHECK(log_now == kKvFlashLogHeader + 10 * RB);
    CHECK(log_now < size_before);
    // the explicit builder, on a head with live records, passes the commit on its own too
    KvFlashSlotHead h;
    std::vector<uint8_t> hb;
    REQUIRE(kvflash_bundle_read(r.store.dir(a) + "/head", hb, HEAD_MAX));
    REQUIRE(kvflash_head_unpack(hb.data(), hb.size(), ID, h));
    KvFlashSlotHead nh;
    const KvFlashBundleCommit c = kvflash_log_compaction_commit(r.store.dir(a), h, RB, nh);
    CHECK(c.new_log);
    CHECK(c.copies.size() == 10);
    CHECK(c.records.empty());
    CHECK(kvflash_bundle_commit(c));
    CHECK(loads(r.store.dir(a)));
}

// ── export and import ───────────────────────────────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, export_copies_exactly_what_the_head_names_and_import_brings_back_identical_tokens_and_meta) {
    Rig r("kvflash_p3_unit_export_import");
    const std::string out = fresh_root("kvflash_p3_unit_export_import_out");
    const std::string a = r.add("sid-a", 100);
    std::ofstream(r.store.dir(a) + "/chunks.9.log") << "a torn leftover";   // not named by the head
    std::ofstream(r.store.dir(a) + "/slab.77") << "a stray slab";
    const KvFlashSessionEntry before = r.ent(a);
    std::vector<int32_t> hist = before.history;
    std::string err;
    REQUIRE(r.st->set_policy(("{\"export_path\":\"" + out + "\", \"after_export\":\"delete\"}").c_str(), err));
    const auto res = r.st->export_session(r.st->claim_of(a), "");   // the policy's path
    REQUIRE(res.status == 200);
    CHECK(names(out + "/" + a) == (std::set<std::string>{"head", "meta.json", kvflash_bundle_log_name(1), kvflash_bundle_slab_name(2)}));
    CHECK(names(out) == (std::set<std::string>{a}));   // no .tmp left
    CHECK(!fs::exists(r.store.dir(a)));                // after_export = delete, after the verified copy
    CHECK(r.store.find_key(a) == nullptr);
    CHECK(loads(out + "/" + a));
    std::string path, state;
    CHECK(r.st->export_result(a, path, state));
    CHECK(path == out + "/" + a);
    CHECK(state == "exported");
    // back in
    std::string key;
    const auto in = r.st->import_dir(out + "/" + a, false, &key);
    REQUIRE(in.status == 200);
    CHECK(key == a);   // its own key is free again
    const KvFlashSessionEntry * e = r.store.find_key(key);
    REQUIRE(e != nullptr);
    CHECK(e->sid == before.sid); CHECK(e->client == before.client); CHECK(e->tokens == before.tokens);
    CHECK(e->created == before.created); CHECK(e->last_used == before.last_used); CHECK(e->gen == before.gen);
    CHECK(e->state == "disk");
    CHECK(e->history == hist);
    std::vector<int32_t> got;
    CHECK(loads(r.store.dir(key), &got));
    CHECK(got == hist);
    CHECK(names(r.store.dir(key)) == (std::set<std::string>{"head", "meta.json", kvflash_bundle_log_name(1), kvflash_bundle_slab_name(2)}));
    // importing the same directory again never overwrites: a second instance under a new key
    std::string key2;
    REQUIRE(r.st->import_dir(out + "/" + a, false, &key2).status == 200);
    CHECK(key2 != key);
    CHECK(loads(r.store.dir(key)));
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, export_with_keep_leaves_the_local_session_and_a_compact_session_exports_its_descriptor) {
    Rig r("kvflash_p3_unit_export_keep", "{\"after_export\":\"keep\"}");
    const std::string out = fresh_root("kvflash_p3_unit_export_keep_out");
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200);
    REQUIRE(r.st->export_session(r.st->claim_of(a), out).status == 200);
    CHECK(fs::exists(r.store.dir(a) + "/head"));
    CHECK(r.ent(a).state == "disk");
    REQUIRE(r.st->compact(r.st->claim_of(b)).status == 200);
    REQUIRE(r.st->export_session(r.st->claim_of(b), out).status == 200);
    CHECK(names(out + "/" + b) == (std::set<std::string>{"head", "meta.json", "compact.json"}));
    // delete locally, import: the compact tier comes back as compact
    REQUIRE(r.st->remove(r.st->claim_of(b)).status == 200);
    std::string key;
    REQUIRE(r.st->import_dir(out + "/" + b, false, &key).status == 200);
    CHECK(r.store.find_key(key)->state == "compact");
    CHECK(!loads(r.store.dir(key)));
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, export_to_a_missing_or_read_only_path_or_over_a_corrupt_log_fails_cleanly_and_keeps_the_session) {
    Rig r("kvflash_p3_unit_export_fail", "{\"after_export\":\"delete\"}");
    const std::string a = r.add("sid-a", 100);
    const std::string missing = "kvflash_p3_unit_no_such_dir/deeper";
    fs::remove_all("kvflash_p3_unit_no_such_dir");
    auto res = r.st->export_session(r.st->claim_of(a), missing);
    CHECK(res.status == 400);
    CHECK(res.error.find("does not exist") != std::string::npos);
    CHECK(!fs::exists("kvflash_p3_unit_no_such_dir"));   // never created for the caller
    CHECK(fs::exists(r.store.dir(a) + "/head"));
    CHECK(r.ent(a).state == "disk");
    CHECK(r.st->export_session(r.st->claim_of(a), "").status == 400);   // no path at all
    // a path that is a file
    std::ofstream("kvflash_p3_unit_export_file") << "x";
    CHECK(r.st->export_session(r.st->claim_of(a), "kvflash_p3_unit_export_file").status == 400);
    fs::remove("kvflash_p3_unit_export_file");
    if (::geteuid() != 0) {   // root ignores directory modes
        const std::string ro = fresh_root("kvflash_p3_unit_export_ro");
        fs::permissions(ro, fs::perms::owner_read | fs::perms::owner_exec);
        res = r.st->export_session(r.st->claim_of(a), ro);
        CHECK(res.status >= 400);
        CHECK(names(ro).empty());
        CHECK(fs::exists(r.store.dir(a) + "/head"));
        fs::permissions(ro, fs::perms::owner_all);
        fs::remove_all(ro);
    }
    // a record that fails its crc: the copy is verified, nothing is deleted, no partial directory is left
    const std::string out = fresh_root("kvflash_p3_unit_export_crc");
    {
        std::fstream f(r.store.dir(a) + "/" + kvflash_bundle_log_name(1), std::ios::in | std::ios::out | std::ios::binary);
        f.seekp((std::streamoff)(kKvFlashLogHeader + kKvFlashRecordTag + 2));
        f.put('\x7f');
    }
    res = r.st->export_session(r.st->claim_of(a), out);
    CHECK(res.status >= 400);
    CHECK(names(out).empty());
    CHECK(fs::exists(r.store.dir(a) + "/head"));
    CHECK(r.store.find_key(a) != nullptr);
    CHECK(r.st->remove(r.st->claim_of(a)).status == 200);   // the claim was released by every failure
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, import_refuses_another_identity_with_the_reason_or_lists_it_foreign_on_request) {
    Rig r("kvflash_p3_unit_import_identity");
    const std::string src = fresh_root("kvflash_p3_unit_import_identity_src");
    {   // a session of another server (identity differs) laid out as an export
        const std::string dir = src + "/their-key";
        park_session(dir, {1, 2, 3, 4}, 2, 0x30, ID + 1);
        KvFlashSessionEntry e;
        e.key = "their-key"; e.sid = "sid-x"; e.client = "header"; e.state = "disk"; e.created = 5; e.last_used = 6; e.tokens = 4; e.identity = ID + 1;
        const std::string m = kvflash_meta_pack(e);
        REQUIRE(kvflash_bundle_write(dir + "/meta.json", std::vector<uint8_t>(m.begin(), m.end())));
    }
    std::string key;
    auto res = r.st->import_dir(src + "/their-key", false, &key);
    CHECK(res.status == 409);
    CHECK(res.error.find("identity") != std::string::npos);
    CHECK(r.store.entries().empty());
    CHECK(names(r.store.dir("")).empty());
    res = r.st->import_dir(src + "/their-key", true, &key);
    CHECK(res.status == 200);
    REQUIRE(r.store.find_key(key) != nullptr);
    CHECK(r.store.find_key(key)->state == "foreign");
    CHECK(!loads(r.store.dir(key)));
    CHECK(r.st->remove(r.st->claim_of(key)).status == 200);   // a foreign directory goes only by an explicit delete
    // not a session directory at all
    fs::create_directories(src + "/empty");
    CHECK(r.st->import_dir(src + "/empty", false, &key).status == 400);
    CHECK(r.st->import_dir(src + "/nope", false, &key).status == 400);
    fs::remove_all(src);
}

// ── the HTTP router (sessions_api.h): shapes and statuses of design section 6 ──────────────────

TEST_CASE(KvflashStoragePolicyFixture, the_router_serves_policy_and_refuses_a_bad_put_naming_the_field) {
    Rig r("kvflash_p3_unit_api_policy");
    auto g = sessions_api_route(*r.st, "GET", "/v1/storage/policy", "");
    CHECK(g.status == 200);
    auto j = nlohmann::json::parse(g.body);
    CHECK(j["action"] == "ask");
    CHECK(j["quota_gib"] == 64);
    CHECK(j["bytes"].contains("reserved_peaks"));
    auto p = sessions_api_route(*r.st, "PUT", "/v1/storage/policy", "{\"quota_gib\": 5}");
    CHECK(p.status == 200);
    CHECK(nlohmann::json::parse(p.body)["quota_gib"] == 5);
    p = sessions_api_route(*r.st, "PUT", "/v1/storage/policy", "{\"quota_gib\": -1}");
    CHECK(p.status == 400);
    CHECK(nlohmann::json::parse(p.body)["error"]["field"] == "quota_gib");
    CHECK(sessions_api_route(*r.st, "POST", "/v1/storage/policy", "").status == 405);
    CHECK(sessions_api_owns("/v1/sessions/abc"));
    CHECK(!sessions_api_owns("/v1/chat/completions"));
}

TEST_CASE(KvflashStoragePolicyFixture, the_router_lists_compacts_and_deletes_a_session_with_the_design_statuses) {
    Rig r("kvflash_p3_unit_api_sessions");
    const std::string key = r.add("api-sid", r.now - 100);
    auto l = nlohmann::json::parse(sessions_api_route(*r.st, "GET", "/v1/sessions", "").body);
    REQUIRE(l.size() == 1);
    CHECK(l[0]["key"] == key);
    CHECK(l[0]["state"] == "disk");
    CHECK(l[0]["restore"]["path"] == "load");
    CHECK(sessions_api_route(*r.st, "GET", "/v1/sessions/" + key, "").status == 200);
    CHECK(sessions_api_route(*r.st, "GET", "/v1/sessions/nope", "").status == 404);
    CHECK(sessions_api_route(*r.st, "GET", "/v1/sessions/..", "").status == 400);
    CHECK(sessions_api_route(*r.st, "POST", "/v1/sessions/" + key + "/compact", "").status == 202);
    CHECK(nlohmann::json::parse(sessions_api_route(*r.st, "GET", "/v1/sessions/" + key, "").body)["state"] == "compact");
    CHECK(sessions_api_route(*r.st, "POST", "/v1/sessions/" + key + "/export", "{}").status == 400);   // no path
    CHECK(sessions_api_route(*r.st, "DELETE", "/v1/sessions/" + key, "").status == 204);
    CHECK(sessions_api_route(*r.st, "DELETE", "/v1/sessions/" + key, "").status == 404);
    CHECK(!fs::exists(r.store.dir(key)));
}

TEST_CASE(KvflashStoragePolicyFixture, the_router_answers_a_pending_decision_and_a_stale_or_unknown_id_is_409_or_404) {
    Rig r("kvflash_p3_unit_api_pending");
    const auto keys = scenario(r);
    r.st->tick();
    auto p = nlohmann::json::parse(sessions_api_route(*r.st, "GET", "/v1/storage/pending", "").body);
    REQUIRE(p.size() >= 1);
    CHECK(p[0]["default"] == "compact");
    CHECK(p[0]["reason"] == "quota");
    const std::string id = p[0]["id"];
    CHECK(sessions_api_route(*r.st, "POST", "/v1/storage/pending/" + id, "{\"action\":\"bogus\"}").status == 400);
    CHECK(sessions_api_route(*r.st, "POST", "/v1/storage/pending/" + id, "{\"action\":\"compact\"}").status == 200);
    CHECK(sessions_api_route(*r.st, "POST", "/v1/storage/pending/" + id, "{\"action\":\"compact\"}").status == 409);   // issued, gone
    CHECK(sessions_api_route(*r.st, "POST", "/v1/storage/pending/p999", "{\"action\":\"compact\"}").status == 404);   // never issued
    CHECK(sessions_api_route(*r.st, "GET", "/v1/storage/preambles", "").body == "[]\n");
    CHECK(sessions_api_route(*r.st, "GET", "/v1/nothing", "").status == 404);
}

// ── code-review regressions (REVIEW_SESSIONS_P0_P3_CODE.md, R05-R17) ────────────────────────────

namespace {
struct Gate {   // holds a worker's I/O until opened
    std::mutex m;
    std::condition_variable cv;
    bool open_ = false;
    void wait() { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&] { return open_; }); }
    void open() { { std::lock_guard<std::mutex> lk(m); open_ = true; } cv.notify_all(); }
};
struct OpenOnExit { Gate & g; ~OpenOnExit() { g.open(); } };
struct Wakes {
    std::atomic<int> n{0};
    bool wait_for(int want, int ms = 5000) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
        while (n.load() < want) {
            if (std::chrono::steady_clock::now() > end) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return true;
    }
};

KvFlashCompactDescriptor descriptor_for(const std::string & dir, const std::vector<int32_t> & hist, uint32_t gen) {
    KvFlashCompactDescriptor d;
    d.gen = gen; d.tokens = hist.size();
    d.history_sha = kvflash_hex16(kvflash_fnv1a(std::string_view((const char *)hist.data(), hist.size() * 4)));
    d.keep = {"head", "meta.json"};
    d.unlink = kvflash_compact_leftovers(dir);
    return d;
}
void write_text(const std::string & path, const std::string & s) { MUST(kvflash_bundle_write(path, std::vector<uint8_t>(s.begin(), s.end()))); }

KvFlashBundleCommit make_commit(const std::string & dir, int n, uint8_t fill) {
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = 1; c.log_gen = 1; c.new_log = true; c.new_dir = true; c.record_bytes = RB;
    KvFlashSlotHead h;
    h.identity = ID; h.gen = 1; h.log_gen = 1; h.end_gen = 2; h.history = {1, 2, 3, 4}; h.pager_hdr = hdr44(); h.ledger.assign((size_t)n * 8, 0);
    for (int i = 0; i < n; ++i) {
        c.records.push_back({(uint32_t)i, std::vector<uint8_t>(CB, (uint8_t)(fill + i))});
        h.map.push_back(kKvFlashLogHeader + (uint64_t)i * RB);
    }
    h.log_bytes = kKvFlashLogHeader + (uint64_t)n * RB;
    c.files = {{kvflash_bundle_slab_name(2), kvflash_slab_pack({{fill, 1}}, {{fill, 2}})}};
    c.head = kvflash_head_pack(h);
    return c;
}
}  // namespace

// R05
TEST_CASE(KvflashStoragePolicyFixture, export_never_lands_inside_the_managed_store_and_never_removes_the_source) {
    Rig r("kvflash_p3_unit_r05");   // after_export = delete: the dangerous setting
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200);
    fs::create_directories(r.store.dir(a) + "/sub");
    fs::create_directories(r.store.dir(b) + "/sub");
    const auto files_a = names(r.store.dir(a)), files_b = names(r.store.dir(b));
    const std::string alias = fresh_root("kvflash_p3_unit_r05_alias");
    fs::create_symlink(fs::absolute(r.store.dir(a)), alias + "/to_a");
    fs::create_symlink(fs::absolute(r.root), alias + "/to_root");
    fs::create_symlink(fs::absolute(r.store.dir(b) + "/sub"), alias + "/to_b_sub");
    const std::vector<std::string> dests = {r.store.dir(a), r.store.dir(a) + "/sub", r.store.dir(b), r.store.dir(b) + "/sub", r.store.dir(""), r.root,
                                            alias + "/to_a", alias + "/to_root", alias + "/to_b_sub"};
    for (const std::string & d : dests) {
        const KvFlashResult res = r.st->export_session(r.st->claim_of(a), d);
        CHECK(res.status == 400);
        CHECK(fs::exists(r.store.dir(a) + "/head"));
        CHECK(loads(r.store.dir(a)));
        CHECK(r.store.find_key(a) != nullptr);
        CHECK(fs::exists(r.store.dir(b) + "/head"));
    }
    CHECK(names(r.store.dir(a)) == files_a);   // nothing was staged or published beneath any of them
    CHECK(names(r.store.dir(b)) == files_b);
    CHECK(names(r.store.dir(a) + "/sub").empty());
    CHECK(names(r.store.dir(b) + "/sub").empty());
    CHECK(r.st->claim_resume(a).status == 200);   // every refusal released its claim
    r.st->release(a);
    fs::remove_all(alias);
}

// R06
TEST_CASE(KvflashStoragePolicyFixture, export_keeps_the_source_and_reports_an_error_when_the_destination_parent_fsync_fails) {
    Rig r("kvflash_p3_unit_r06");
    const std::string out = fresh_root("kvflash_p3_unit_r06_out");
    const std::string out_c = fs::weakly_canonical(out).string();
    r.env.fsync_dir = [out_c](const std::string & d) { return fs::weakly_canonical(d).string() != out_c && kvflash_fsync_dir(d); };
    r.start("{}");   // after_export = delete
    const std::string a = r.add("sid-a", 100);
    const KvFlashResult res = r.st->export_session(r.st->claim_of(a), out);
    CHECK(!res.ok());
    CHECK(res.status == 500);
    CHECK(res.error.find("fsync") != std::string::npos);
    CHECK(fs::exists(r.store.dir(a) + "/head"));
    CHECK(loads(r.store.dir(a)));
    CHECK(r.store.find_key(a) != nullptr);
    std::string path, state;
    CHECK(r.st->export_result(a, path, state));
    CHECK(state == "failed");
    CHECK(r.st->claim_resume(a).status == 200);
    r.st->release(a);
    fs::remove_all(out);
}

// R07
TEST_CASE(KvflashStoragePolicyFixture, a_failed_compaction_unlink_never_reaches_a_later_instance_under_the_same_key) {
    for (const bool by_export : {false, true}) {
        Rig r("kvflash_p3_unit_r07", "{\"after_export\":\"keep\"}");
        const std::string out = fresh_root("kvflash_p3_unit_r07_out"), out2 = fresh_root("kvflash_p3_unit_r07_out2");
        const std::string a = r.add("sid-a", 100);
        REQUIRE(r.st->export_session(r.st->claim_of(a), out).status == 200);   // the retained external full export
        r.unlink_fails = {kvflash_bundle_log_name(1)};
        const KvFlashResult c = r.st->compact(r.st->claim_of(a));
        REQUIRE(c.status == 200);
        REQUIRE(!c.error.empty());                 // the unlink debt
        r.unlink_fails.clear();
        if (by_export) {
            std::string err;
            REQUIRE(r.st->set_policy("{\"after_export\":\"delete\"}", err));
            REQUIRE(r.st->export_session(r.st->claim_of(a), out2).status == 200);   // the compact copy leaves, the local one is deleted
        } else {
            REQUIRE(r.st->remove(r.st->claim_of(a)).status == 200);
        }
        REQUIRE(r.store.find_key(a) == nullptr);
        std::string key;
        REQUIRE(r.st->import_dir(out + "/" + a, false, &key).status == 200);
        CHECK(key == a);                            // the key is reused
        r.st->tick();
        CHECK(loads(r.store.dir(a)));               // the new instance keeps its reconstruction files
        CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
        CHECK(r.store.find_key(a)->state == "disk");
        fs::remove_all(out); fs::remove_all(out2);
    }
}

// R07 + R11: the retry cleans only what a valid descriptor of the current head names
TEST_CASE(KvflashStoragePolicyFixture, the_unlink_retry_needs_a_valid_descriptor_and_removes_only_the_files_it_names) {
    Rig r("kvflash_p3_unit_r07_valid");
    const std::string a = r.add("sid-a", 100);
    const std::vector<int32_t> hist = r.ent(a).history;
    r.unlink_fails = {kvflash_bundle_log_name(1)};
    REQUIRE(r.st->compact(r.st->claim_of(a)).status == 200);
    r.unlink_fails.clear();
    const std::string good = slurp(r.store.dir(a) + "/compact.json");
    const std::string log = r.store.dir(a) + "/" + kvflash_bundle_log_name(1);
    write_text(r.store.dir(a) + "/compact.json", "");   // damaged
    r.st->tick();
    CHECK(fs::exists(log));
    KvFlashCompactDescriptor d = descriptor_for(r.store.dir(a), hist, 1);
    d.unlink.push_back("head");                          // a descriptor that names the head is not trusted
    const std::string bad = kvflash_compact_pack(d);
    write_text(r.store.dir(a) + "/compact.json", bad);
    r.st->tick();
    CHECK(fs::exists(log));
    CHECK(fs::exists(r.store.dir(a) + "/head"));
    write_text(r.store.dir(a) + "/compact.json", good);
    r.st->tick();
    CHECK(!fs::exists(log));
    CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json"}));
}

// R11
TEST_CASE(KvflashStoragePolicyFixture, a_descriptor_that_does_not_match_the_head_never_authorizes_cleanup_and_the_record_stays_foreign) {
    for (const std::string kind : {"empty", "malformed", "gen", "sha", "tokens", "other_head", "unsafe", "valid"}) {
        Rig r("kvflash_p3_unit_r11");
        const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200);
        const std::vector<int32_t> hist = r.ent(a).history, hist_b = r.ent(b).history;
        KvFlashCompactDescriptor d = descriptor_for(r.store.dir(a), hist, 1);
        if (kind == "empty") write_text(r.store.dir(a) + "/compact.json", "");
        else if (kind == "malformed") write_text(r.store.dir(a) + "/compact.json", "{not json");
        else {
            if (kind == "gen") d.gen = 7;
            if (kind == "sha") d.history_sha = "0000000000000000";
            if (kind == "tokens") d.tokens += 1;
            if (kind == "other_head") d = descriptor_for(r.store.dir(b), hist_b, 1);
            if (kind == "unsafe") d.unlink.push_back("../" + b + "/head");
            write_text(r.store.dir(a) + "/compact.json", kvflash_compact_pack(d));
        }
        const auto es = r.rescan();
        const KvFlashSessionEntry * e = find(es, a);
        REQUIRE(e != nullptr);
        if (kind == "valid") {
            CHECK(e->state == "compact");
            CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json"}));
        } else {
            CHECK(e->state == "foreign");
            CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
            CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_slab_name(2)));
            CHECK(loads(r.store.dir(a)));
        }
        CHECK(fs::exists(r.store.dir(b) + "/head"));
        CHECK(find(es, b)->state == "disk");
    }
}

TEST_CASE(KvflashStoragePolicyFixture, import_and_compact_export_refuse_a_descriptor_that_does_not_match_its_head) {
    Rig r("kvflash_p3_unit_r11_import", "{\"after_export\":\"keep\"}");
    const std::string out = fresh_root("kvflash_p3_unit_r11_import_out");
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200);
    REQUIRE(r.st->export_session(r.st->claim_of(a), out).status == 200);
    write_text(out + "/" + a + "/compact.json", "{garbage");   // a marker in a full export
    std::string key;
    auto res = r.st->import_dir(out + "/" + a, false, &key);
    CHECK(!res.ok());
    CHECK(names(r.store.dir("")).count(key) == 0);
    CHECK(r.store.entries().size() == 2);
    res = r.st->import_dir(out + "/" + a, true, &key);   // an operator can still keep it as a foreign record, assets intact
    REQUIRE(res.ok());
    CHECK(r.store.find_key(key)->state == "foreign");
    CHECK(fs::exists(r.store.dir(key) + "/" + kvflash_bundle_log_name(1)));
    // a compact session whose descriptor was damaged cannot be exported as compact
    REQUIRE(r.st->compact(r.st->claim_of(b)).status == 200);
    write_text(r.store.dir(b) + "/compact.json", "");
    res = r.st->export_session(r.st->claim_of(b), out);
    CHECK(!res.ok());
    CHECK(!fs::exists(out + "/" + b));
    CHECK(r.st->claim_resume(b).status == 200);   // claim released
    r.st->release(b);
    fs::remove_all(out);
}

// R12
TEST_CASE(KvflashStoragePolicyFixture, a_reservation_is_the_bytes_of_the_generated_commit_with_tags_header_head_meta_and_index) {
    Rig r("kvflash_p3_unit_r12_bytes", "{\"reserve_gib\": 1}");
    const std::string dir = r.store.dir("k-new");
    const KvFlashBundleCommit c = make_commit(dir, 6, 0x50);
    const uint64_t predicted = kvflash_commit_bytes(c);
    REQUIRE(kvflash_bundle_commit(c));
    CHECK(predicted == tree_bytes(dir));          // header + (payload + 16-byte tag) per record + slabs + head, exactly
    CHECK(predicted == kKvFlashLogHeader + 6 * RB + c.files[0].second.size() + c.head.size());
    KvFlashSessionEntry e;
    e.key = "k-new"; e.sid = "sid-new"; e.client = "header"; e.state = "parked-gpu";
    uint64_t id = 0;
    std::string why;
    r.free_b = 1 * GIB + (1ull << 30);
    REQUIRE(r.st->reserve_commit(c, e, id, why));
    const uint64_t peak = r.st->occupancy().reserved_peaks;
    CHECK(peak >= predicted + kvflash_meta_pack(e).size());   // the meta and index replacements are on top
    CHECK(peak > predicted);
    r.st->release_reservation(id);
    r.free_b = 1 * GIB + peak - 1;
    CHECK(!r.st->reserve_commit(c, e, id, why));
    r.free_b = 1 * GIB + peak;
    CHECK(r.st->reserve_commit(c, e, id, why));
    CHECK(r.st->occupancy().reserved_peaks == peak);
}

TEST_CASE(KvflashStoragePolicyFixture, queued_writes_together_may_not_cross_the_quota_growth_limit) {
    Rig r("kvflash_p3_unit_r12_quota", "{\"quota_gib\": 4, \"reserve_gib\": 0}");   // quota 256 bytes, hard limit 1.1 x
    uint64_t i1 = 0, i2 = 0;
    std::string why;
    CHECK(r.st->reserve(150, i1, why));
    CHECK(!r.st->reserve(150, i2, why));   // 300 > 281.6: each alone fits, the queue does not
    CHECK(why.find("quota") != std::string::npos);
    r.st->release_reservation(i1);
    CHECK(r.st->reserve(150, i2, why));
}

TEST_CASE(KvflashStoragePolicyFixture, policy_numbers_are_bounded_so_no_conversion_overflows) {
    Rig r("kvflash_p3_unit_r12_bounds");
    for (const char * body : {"{\"quota_gib\":17179869184}", "{\"reserve_gib\":17179869184}", "{\"max_age_days\":18446744073709551615}",
                              "{\"grace_hours\":18446744073709551615}", "{\"log_dead_mib\":18446744073709551615}", "{\"preamble_mib\":18446744073709551615}",
                              "{\"quota_gib\":\"5\"}", "{\"quota_gib\":5.5}", "{\"quota_gib\":-1}", "{\"quota_gib\":null}", "{\"quota_gib\":true}",
                              "{\"action\":1}", "{\"action\":null}", "{\"export_path\":3}", "{\"after_export\":[]}", "[1]", "{\"quota_gib\":5} trailing"}) {
        std::string err;
        CHECK(!r.st->set_policy(body, err));
        CHECK(!err.empty());
        CHECK(r.st->policy().quota_gib == 64);
    }
    std::string err;
    CHECK(r.st->set_policy("{\"quota_gib\":1048576, \"max_age_days\":36500}", err));
    CHECK(r.st->occupancy().quota == 1048576ull * GIB);
}

// R08
TEST_CASE(KvflashStoragePolicyFixture, wrong_field_types_nulls_and_bad_encodings_are_400_and_never_throw) {
    Rig r("kvflash_p3_unit_r08");
    const std::string a = r.add("sid-a", 100);
    bool threw = false;
    auto call = [&](const std::string & method, const std::string & path, const std::string & body) {
        try { return sessions_api_route(*r.st, method, path, body); }
        catch (...) { threw = true; return SessionsApiReply{500, ""}; }
    };
    for (const char * body : {"{\"path\":1}", "{\"path\":null}", "{\"path\":[]}", "{\"path\":true}", "{\"path\":{\"a\":1}}", "{\"path\":\"a\\u0000b\"}"})
        CHECK(call("POST", "/v1/sessions/" + a + "/export", body).status == 400);
    scenario(r);
    r.st->tick();
    REQUIRE(!r.st->pending().empty());
    const std::string id = r.st->pending()[0].id;
    for (const char * body : {"{\"action\":\"export\",\"path\":5}", "{\"action\":\"export\",\"path\":null}", "{\"action\":3}", "{\"action\":null}"})
        CHECK(call("POST", "/v1/storage/pending/" + id, body).status == 400);
    for (const char * body : {"{\"path\":7}", "{\"path\":\"x\",\"foreign\":\"yes\"}", "{\"path\":\"x\",\"foreign\":1}", "{\"path\":\"x\",\"foreign\":null}", "[]", "{\"path\":\"\\ud800\"}"})
        CHECK(call("POST", "/v1/sessions/import", body).status == 400);
    CHECK(call("POST", "/v1/sessions/import", std::string("{\"path\":\"\xff\xfe\"}")).status == 400);
    CHECK(call("PUT", "/v1/storage/policy", std::string("{\"export_path\":\"\xff\"}")).status == 400);
    CHECK(call("PUT", "/v1/storage/policy", "{\"export_path\":\"\\ud800\"}").status == 400);   // a lone surrogate
    // a valid escaped surrogate pair is one supplementary character, and the policy answer parses back
    CHECK(call("PUT", "/v1/storage/policy", "{\"export_path\":\"/exports/\\ud83d\\ude00\"}").status == 200);
    const auto g = call("GET", "/v1/storage/policy", "");
    CHECK(g.status == 200);
    CHECK(nlohmann::json::parse(g.body)["export_path"] == "/exports/\xf0\x9f\x98\x80");
    // an identity that is not UTF-8 (a foreign meta.json) lists without throwing
    r.ent(a).sid = "bad\xff";
    CHECK(call("GET", "/v1/sessions", "").status == 200);
    CHECK(call("GET", "/v1/sessions/" + a, "").status == 200);
    CHECK(!threw);
}

// R16
TEST_CASE(KvflashStoragePolicyFixture, a_quarantined_record_can_be_listed_inspected_and_deleted_by_its_generated_key) {
    Rig r("kvflash_p3_unit_r16");
    fs::create_directories(r.root + "/slot0");
    park_session(r.root + "/slot0", {1, 2, 3}, 2, 0x40, ID + 1);   // another server's head: the migration sets it aside
    KvFlashScanReport rep;
    kvflash_migrate_slot_dirs(r.root, ID, HEAD_MAX, rep);
    const std::string qkey = "slot0.quarantine.1";
    REQUIRE(fs::exists(r.store.dir(qkey) + "/head"));
    r.store.entries() = r.rescan();
    const auto l = nlohmann::json::parse(sessions_api_route(*r.st, "GET", "/v1/sessions", "").body);
    REQUIRE(l.size() == 1);
    CHECK(l[0]["key"] == qkey);
    CHECK(l[0]["state"] == "foreign");
    const auto g = sessions_api_route(*r.st, "GET", "/v1/sessions/" + qkey, "");
    CHECK(g.status == 200);
    for (const char * bad : {"..", ".", "a..b", "slot0.quarantine.1..", "slot0.quarantine.", "slot0.quarantine.1.2", "x%2f", "x\\y", ".hidden", "slot0.quarantine.1%00"})
        CHECK(sessions_api_route(*r.st, "GET", std::string("/v1/sessions/") + bad, "").status == 400);
    CHECK(sessions_api_route(*r.st, "GET", "/v1/sessions/slot0.quarantine.99", "").status == 404);   // well-formed, unknown
    CHECK(sessions_api_route(*r.st, "DELETE", "/v1/sessions/slot0.quarantine.99", "").status == 404);
    CHECK(sessions_api_route(*r.st, "DELETE", "/v1/sessions/" + qkey, "").status == 204);
    CHECK(!fs::exists(r.store.dir(qkey)));
    CHECK(r.store.find_key(qkey) == nullptr);
}

// R10 + R17: the asynchronous export, through the worker, and the next tick
TEST_CASE(KvflashStoragePolicyFixture, an_asynchronous_export_wakes_the_owner_and_its_claim_is_released_at_the_next_pump) {
    Rig r("kvflash_p3_unit_r10_export", "{}");
    r.env.async_export = true;
    r.env.tick_interval_s = 3600;
    r.start("{\"after_export\":\"keep\"}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string out = fresh_root("kvflash_p3_unit_r10_export_out");
    const std::string a = r.add("sid-a", 100);
    REQUIRE(r.st->export_async(r.st->claim_of(a), out).status == 202);
    CHECK(w.wait_for(1));                                    // the worker wakes the scheduler: no traffic needed
    CHECK(r.st->remove(r.st->claim_of(a)).status == 409);    // the claim is held until the owner commits
    r.st->pump();
    std::string path, state;
    CHECK(r.st->export_result(a, path, state));
    CHECK(state == "copied");
    CHECK(r.st->claim_resume(a).status == 200);              // released
    r.st->release(a);
    r.st->set_wake({});
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, a_pending_decision_is_acted_on_at_its_deadline_without_waiting_for_the_next_interval) {
    Rig r("kvflash_p3_unit_r10_deadline");
    r.env.grace_s = 60;
    r.env.tick_interval_s = 3600;
    r.start("{}");
    const auto keys = scenario(r);
    CHECK(r.st->wake_ms() == 0);                             // a first pass is due at once
    r.st->pump();                                            // the first pass offers the decisions
    REQUIRE(r.st->pending().size() == 2);
    const int64_t wake = r.st->wake_ms();
    CHECK(wake > 0);
    CHECK(wake <= 60 * 1000);                                // the scheduler wakes for the deadline, not the interval
    r.now += 30;
    r.st->pump();
    CHECK(r.ent(keys[0]).state == "disk");
    CHECK(r.st->wake_ms() <= 30 * 1000);
    r.now += 31;
    CHECK(r.st->wake_ms() == 0);
    r.st->pump();                                            // inside the 3600 s interval
    CHECK(r.ent(keys[0]).state == "compact");
}

// R13
TEST_CASE(KvflashStoragePolicyFixture, a_log_compaction_copies_on_a_worker_while_the_tick_returns_and_the_owner_commits_it) {
    Gate gate;
    Rig r("kvflash_p3_unit_r13_log", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"reserve_gib\": 1, \"quota_gib\": 1000, \"log_dead_mib\": 1, \"action\":\"compact\"}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string a = r.add("sid-a", r.now - 1000, 40, "disk", /*live*/ 10);
    r.ent(a).bytes_dead = 30 * RB;
    r.st->tick();                                            // plans the rewrite; the copy waits at the gate
    CHECK(r.ent(a).gen == 1);                                // nothing was committed on the tick
    CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(r.st->remove(r.st->claim_of(a)).status == 409);    // the claim is held for the worker
    gate.open();
    CHECK(w.wait_for(1));
    r.st->pump();                                            // the owner thread publishes the result
    CHECK(r.ent(a).gen > 1);
    CHECK(r.ent(a).bytes_dead == 0);
    CHECK(loads(r.store.dir(a)));
    CHECK(!fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(r.st->claim_resume(a).status == 200);
    r.st->release(a);
    r.st->set_wake({});
}

TEST_CASE(KvflashStoragePolicyFixture, an_import_verifies_and_copies_on_a_worker_and_is_listed_importing_until_the_owner_commits) {
    Gate gate;
    Rig r("kvflash_p3_unit_r13_import", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string out = fresh_root("kvflash_p3_unit_r13_import_out");
    const std::string a = r.add("sid-a", 100);
    fs::copy(r.store.dir(a), out + "/a1", fs::copy_options::recursive);
    fs::copy(r.store.dir(a), out + "/a2", fs::copy_options::recursive);
    std::string k1, k2;
    CHECK(r.st->import_dir(out + "/a1", false, &k1).status == 202);
    CHECK(r.st->import_dir(out + "/a2", false, &k2).status == 202);
    CHECK(!k1.empty());
    CHECK(k1 != k2);                                         // the second import cannot take the first's key
    CHECK(r.store.find_key(k1) == nullptr);
    std::string st, err;
    REQUIRE(r.st->import_result(k1, st, &err));
    CHECK(st == "importing");
    const auto g = sessions_api_route(*r.st, "GET", "/v1/sessions/" + k1, "");
    CHECK(g.status == 200);
    CHECK(nlohmann::json::parse(g.body)["state"] == "importing");
    gate.open();
    CHECK(w.wait_for(2));
    r.st->pump();
    REQUIRE(r.store.find_key(k1) != nullptr);
    REQUIRE(r.store.find_key(k2) != nullptr);
    CHECK(r.store.find_key(k1)->state == "disk");
    CHECK(loads(r.store.dir(k1)));
    CHECK(loads(r.store.dir(k2)));
    CHECK(r.st->import_result(k1, st, &err));
    CHECK(st == "imported");
    r.st->set_wake({});
    fs::remove_all(out);
}

// R14
TEST_CASE(KvflashStoragePolicyFixture, a_generation_tag_is_published_only_after_the_writer_acknowledges_it) {
    Rig r("kvflash_p3_unit_r14_ack");
    const std::string a = r.add("sid-a", 100, 40, "parked-gpu");
    r.st->writer_begin(a);
    r.st->writer_ack(a, 2, true);
    CHECK(r.st->claim_resume(a).status == 409);              // held until the owner thread has reconciled the ack
    CHECK(r.ent(a).gen == 1);
    r.st->apply_acks();
    CHECK(r.ent(a).gen == 2);
    CHECK(slurp(r.store.dir(a) + "/meta.json").find("\"gen\":2") != std::string::npos);
    CHECK(r.st->claim_resume(a).status == 200);
    r.st->release(a);
    r.st->writer_begin(a);
    r.st->writer_ack(a, 3, false);                           // the writer failed: the tag is never published
    r.st->apply_acks();
    CHECK(r.ent(a).gen == 2);
    r.st->writer_begin(a);
    r.st->writer_ack(a, 1, true);                            // and a late lower ack cannot roll it back
    r.st->apply_acks();
    CHECK(r.ent(a).gen == 2);
}

TEST_CASE(KvflashStoragePolicyFixture, an_import_normalizes_its_metadata_before_publishing_and_a_publish_fsync_failure_leaves_nothing) {
    Rig r("kvflash_p3_unit_r14_import");
    const std::string out = fresh_root("kvflash_p3_unit_r14_import_out");
    const std::string a = r.add("sid-a", 100);
    fs::copy(r.store.dir(a), out + "/a", fs::copy_options::recursive);
    KvFlashSessionEntry old = r.ent(a);                      // the export's meta names archive relations of the old server
    old.archive_of = "old-key"; old.archive = "succ-key"; old.state = "exported";
    write_text(out + "/a/meta.json", kvflash_meta_pack(old));
    std::string seen_meta;
    bool fail_publish = false;
    const std::string sessions_c = fs::weakly_canonical(r.store.dir("")).string();
    r.env.fsync_dir = [&](const std::string & d) {
        if (d.size() > 4 && d.compare(d.size() - 4, 4, ".tmp") == 0) seen_meta = slurp(d + "/meta.json");   // what is about to be published
        if (fail_publish && fs::weakly_canonical(d).string() == sessions_c) return false;
        return kvflash_fsync_dir(d);
    };
    r.start("{}");
    std::string key;
    REQUIRE(r.st->import_dir(out + "/a", false, &key).status == 200);
    KvFlashSessionEntry got;
    REQUIRE(kvflash_meta_unpack(seen_meta, got));
    CHECK(got.archive_of.empty());
    CHECK(got.archive.empty());
    CHECK(got.state == "disk");
    CHECK(r.store.find_key(key)->archive.empty());
    // a failed fsync of the sessions directory after the rename: the import fails and publishes nothing
    fail_publish = true;
    const size_t n = r.store.entries().size();
    const auto before = names(r.store.dir(""));
    const auto res = r.st->import_dir(out + "/a", false, &key);
    CHECK(res.status == 500);
    CHECK(r.store.entries().size() == n);
    CHECK(names(r.store.dir("")) == before);
    fs::remove_all(out);
}

// R17: the engine's park decision and the restore claim, through the functions the engine calls
TEST_CASE(KvflashStoragePolicyFixture, a_refused_park_marks_the_slot_for_a_rewrite_and_the_session_stays_in_its_slot_under_pressure) {
    Rig r("kvflash_p3_unit_r17_park", "{\"reserve_gib\": 4, \"quota_gib\": 1, \"action\":\"delete-oldest\"}");
    const std::string owner = r.add("sid-owner", 100, 40, "parked-gpu");
    const KvFlashBundleCommit c = make_commit(r.store.dir(owner), 6, 0x60);
    KvFlashSessionEntry e = r.ent(owner);
    KvFlashSlotBundle sb = KvFlashSlotBundle::adopted(owner, "sid-owner", "header", 90, KvFlashSlotHead{});
    sb.head.gen = 1;
    r.free_b = 4 * GIB + 10;
    uint64_t res = 0;
    std::string why;
    CHECK(!kvflash_park_admit(r.st.get(), c, e, sb, res, why));
    CHECK(sb.rewrite);                                       // the next park rewrites from the acknowledged head
    CHECK(!sb.pending);
    CHECK(res == 0);
    CHECK(r.st->occupancy().reserved_peaks == 0);
    CHECK(!kvflash_session_park_durable(sb.pending, false, sb.rewrite, sb.head.gen));   // so the slot is not evicted to disk
    for (int i = 0; i < 3; ++i) { r.st->tick(); r.now += 100 * 3600; }
    CHECK(r.ent(owner).state == "parked-gpu");               // and no policy discards it
    CHECK(fs::exists(r.store.dir(owner) + "/head"));
    r.free_b = 1ull << 40;
    sb.rewrite = false;
    CHECK(kvflash_park_admit(r.st.get(), c, e, sb, res, why));
    CHECK(res != 0);
    CHECK(r.st->occupancy().reserved_peaks >= kvflash_commit_bytes(c));
    CHECK(kvflash_park_admit(nullptr, c, e, sb, res, why));  // no storage: nothing to ask
}
