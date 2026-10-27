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
#include <sys/stat.h>
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
    bool ok = false;   // a worker may be deleting beneath it: wait it out; a short count is a failed test, never a smaller number
    for (int i = 0; i < 200 && !(ok = kvflash_tree_bytes_checked(dir, n)); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    MUST(ok);
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
    CHECK(p.reserve_gib < 0);   // auto: min(10% of the disk, 4 GiB), kvflash_auto_reserve_bytes
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

TEST_CASE(KvflashStoragePolicyFixture, an_instance_archived_at_admission_is_exempt_before_the_successors_first_park_publishes_the_link) {
    Rig r("kvflash_p3_unit_exempt_admission", "{\"quota_gib\": 1000, \"max_age_days\": 30}");
    const std::string old = r.add("sid-a", r.now - 90 * 86400);
    r.st->tick();
    REQUIRE(r.st->pending().size() == 1);                              // an ordinary old disk session: offered
    // the admission evicted `old` and listed its successor (archive_of set); `old` does not name it yet
    const std::string succ = r.add("sid-a#2", r.now - 10, 40, "parked-gpu");
    r.ent(succ).archive_of = old;
    CHECK(r.ent(old).archive.empty());
    r.st->tick();
    CHECK(r.st->pending().empty());                                    // the standing decision is void, nothing new is offered
    CHECK(fs::exists(r.store.dir(old) + "/head"));
    r.now += 100 * 86400;                                              // far past any grace: still nothing happens to it
    r.st->tick();
    CHECK(r.ent(old).state == "disk");
    CHECK(fs::exists(r.store.dir(old) + "/head"));
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
    CHECK(fs::exists(out + "/" + a + ".g1" + "/head"));
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
    Rig r("kvflash_p3_unit_reserve_first_park", "{\"reserve_gib\": 4, \"quota_gib\": 1000}");   // 4 GiB reserve in test units
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
    Rig r("kvflash_p3_unit_export_import", "{\"quota_gib\": 1000}");
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
    CHECK(names(out + "/" + a + ".g1") == (std::set<std::string>{"head", "meta.json", kvflash_bundle_log_name(1), kvflash_bundle_slab_name(2)}));
    CHECK(names(out) == (std::set<std::string>{a + ".g1"}));   // no .tmp left
    CHECK(!fs::exists(r.store.dir(a)));                // after_export = delete, after the verified copy
    CHECK(r.store.find_key(a) == nullptr);
    CHECK(loads(out + "/" + a + ".g1"));
    std::string path, state;
    CHECK(r.st->export_result(a, path, state));
    CHECK(path == out + "/" + a + ".g1");
    CHECK(state == "exported");
    // back in
    std::string key;
    const auto in = r.st->import_dir(out + "/" + a + ".g1", false, &key);
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
    REQUIRE(r.st->import_dir(out + "/" + a + ".g1", false, &key2).status == 200);
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
    CHECK(names(out + "/" + b + ".g1") == (std::set<std::string>{"head", "meta.json", "compact.json"}));
    // delete locally, import: the compact tier comes back as compact
    REQUIRE(r.st->remove(r.st->claim_of(b)).status == 200);
    std::string key;
    REQUIRE(r.st->import_dir(out + "/" + b + ".g1", false, &key).status == 200);
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
// F04: the staging directory is checked like the destination, never erased unless the export made it, and the destination is held by descriptor.
TEST_CASE(KvflashStoragePolicyFixture, an_export_whose_staging_name_is_the_managed_root_lands_beside_it_and_the_store_survives) {
    const std::string out = fresh_root("kvflash_p3_unit_f04_collide_out");
    const std::string key = kvflash_sid_key("header", "sid-a");
    Rig r(out + "/" + key + ".g1.tmp");   // the managed root is D/<key>.tmp: exactly where the export would stage session <key>
    const std::string a = r.add("sid-a", 100);
    REQUIRE(a == key);
    const KvFlashResult res = r.st->export_session(r.st->claim_of(a), out);   // the name is taken: the export lands beside it as <key>.g1.1
    CHECK(res.ok());
    CHECK(fs::exists(r.root + "/policy.json"));   // the managed root was never staged into, nor erased
    CHECK(loads(out + "/" + a + ".g1.1"));
}

TEST_CASE(KvflashStoragePolicyFixture, an_export_never_removes_a_staging_directory_it_did_not_create) {
    Rig r("kvflash_p3_unit_f04_stale");
    const std::string out = fresh_root("kvflash_p3_unit_f04_stale_out");
    const std::string a = r.add("sid-a", 100);
    fs::create_directories(out + "/" + a + ".g1.tmp/inner");
    write_text(out + "/" + a + ".g1.tmp/inner/keep.txt", "not the export's");
    const KvFlashResult res = r.st->export_session(r.st->claim_of(a), out);   // the staging name is taken: the export uses the next one
    CHECK(res.ok());
    CHECK(slurp(out + "/" + a + ".g1.tmp/inner/keep.txt") == "not the export's");
    CHECK(loads(out + "/" + a + ".g1.1"));
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, an_export_lands_in_the_destination_it_was_planned_against_even_if_its_pathname_is_replaced_afterwards) {
    Rig r("kvflash_p3_unit_f04_toctou");
    const std::string out = fresh_root("kvflash_p3_unit_f04_toctou_out");
    const std::string a = r.add("sid-a", 100);
    KvFlashExportPlan plan;
    REQUIRE(r.st->export_begin(r.st->claim_of(a), out, plan).ok());
    // after the plan: the resolved destination is moved away and its pathname becomes a link into the source session
    const std::string moved = out + "_moved";
    fs::remove_all(moved);
    fs::rename(out, moved);
    fs::create_directory_symlink(fs::absolute(r.store.dir(a)), out);
    const KvFlashResult copied = r.st->export_copy(plan);
    CHECK(copied.ok());
    CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", kvflash_bundle_log_name(1), kvflash_bundle_slab_name(2)}));   // nothing landed beneath the source
    CHECK(loads(moved + "/" + a + ".g1"));                                                                                                           // it landed where it was planned
    CHECK(r.st->export_commit(plan, copied).ok());                                                                                           // and only then is the source removed
    CHECK(!fs::exists(r.store.dir(a)));
    CHECK(loads(moved + "/" + a + ".g1"));
    fs::remove(out);
    fs::remove_all(moved);
}

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
        REQUIRE(r.st->import_dir(out + "/" + a + ".g1", false, &key).status == 200);
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
    for (const std::string kind : {"empty", "malformed", "gen", "sha", "tokens", "other_head", "unsafe", "trailing", "trailing_brace", "wrap_gen", "junk_number", "valid"}) {
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
            std::string text = kvflash_compact_pack(d);
            auto swap = [&](const std::string & from, const std::string & to) { const size_t at = text.find(from); MUST(at != std::string::npos); text.replace(at, from.size(), to); };
            if (kind == "trailing") text += "garbage";                                      // F05: a matching descriptor with a tail is no descriptor
            if (kind == "trailing_brace") text += "}";
            if (kind == "wrap_gen") swap("\"gen\":1,", "\"gen\":4294967297,");           // 2^32 + 1 must not wrap to the head's generation 1
            if (kind == "junk_number") swap("\"tokens\":" + std::to_string(d.tokens) + ",", "\"tokens\":" + std::to_string(d.tokens) + "abc,");
            write_text(r.store.dir(a) + "/compact.json", text);
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

TEST_CASE(KvflashStoragePolicyFixture, a_flat_json_object_with_anything_after_its_closing_brace_is_refused) {
    std::map<std::string, std::string> m;
    CHECK(detail::json_flat_object("{\"a\":\"b\"}", m));
    CHECK(detail::json_flat_object("  {\"a\":1, \"b\":\"c\"}\n", m));
    CHECK(detail::json_flat_object("{}", m));
    CHECK(!detail::json_flat_object("{\"a\":\"b\"}x", m));
    CHECK(!detail::json_flat_object("{\"a\":\"b\"} {\"c\":\"d\"}", m));
    CHECK(!detail::json_flat_object("{} junk", m));
    KvFlashSessionEntry e;
    e.sid = "s"; e.client = "header";
    CHECK(kvflash_meta_unpack(kvflash_meta_pack(e), e));
    CHECK(!kvflash_meta_unpack(kvflash_meta_pack(e) + "junk", e));
}

// F08: an import is admitted only if the bundle would restore: a slab that exists but is truncated is refused, nothing is published.
TEST_CASE(KvflashStoragePolicyFixture, an_import_whose_slab_is_present_but_malformed_is_refused_and_leaves_nothing_published) {
    Rig r("kvflash_p3_unit_f08_import", "{\"after_export\":\"keep\", \"quota_gib\": 1000}");
    const std::string out = fresh_root("kvflash_p3_unit_f08_import_out");
    const std::string a = r.add("sid-a", 100);
    REQUIRE(r.st->export_session(r.st->claim_of(a), out).status == 200);
    const std::string slab = out + "/" + a + ".g1/" + kvflash_bundle_slab_name(2);
    REQUIRE(fs::exists(slab));
    fs::resize_file(slab, 3);   // present, not a slab
    const size_t before = r.store.entries().size();
    std::string key;
    const KvFlashResult res = r.st->import_dir(out + "/" + a + ".g1", false, &key);
    CHECK(!res.ok());
    CHECK(res.error.find("slab") != std::string::npos);
    CHECK(r.store.entries().size() == before);
    CHECK(!fs::exists(r.store.dir(key)));
    CHECK(!fs::exists(r.store.dir(key) + ".tmp"));
    std::string state;
    CHECK(r.st->import_result(key, state));
    CHECK(state == "failed");
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, import_and_compact_export_refuse_a_descriptor_that_does_not_match_its_head) {
    Rig r("kvflash_p3_unit_r11_import", "{\"after_export\":\"keep\", \"quota_gib\": 1000}");
    const std::string out = fresh_root("kvflash_p3_unit_r11_import_out");
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 200);
    REQUIRE(r.st->export_session(r.st->claim_of(a), out).status == 200);
    write_text(out + "/" + a + ".g1/compact.json", "{garbage");   // a marker in a full export
    std::string key;
    auto res = r.st->import_dir(out + "/" + a + ".g1", false, &key);
    CHECK(!res.ok());
    CHECK(names(r.store.dir("")).count(key) == 0);
    CHECK(r.store.entries().size() == 2);
    res = r.st->import_dir(out + "/" + a + ".g1", true, &key);   // an operator can still keep it as a foreign record, assets intact
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
    Rig r("kvflash_p3_unit_r12_quota", "{\"quota_gib\": 8, \"reserve_gib\": 0}");   // quota 512 bytes, hard limit 563 (the store holds policy.json)
    uint64_t i1 = 0, i2 = 0;
    std::string why;
    CHECK(r.st->reserve(300, i1, why));
    CHECK(!r.st->reserve(300, i2, why));   // each alone fits, the queue does not
    CHECK(why.find("quota") != std::string::npos);
    r.st->release_reservation(i1);
    CHECK(r.st->reserve(300, i2, why));
}

// F06: the projected growth is checked for the first reservation too, not only once something is queued.
TEST_CASE(KvflashStoragePolicyFixture, a_single_write_that_passes_the_quota_growth_limit_is_refused_with_nothing_queued) {
    Rig r("kvflash_p3_unit_f06_quota", "{\"quota_gib\": 8, \"reserve_gib\": 0}");   // quota 512 bytes, hard limit 563
    uint64_t id = 0;
    std::string why;
    CHECK(!r.st->reserve(600, id, why));   // alone, nothing queued: the store and this write pass the limit
    CHECK(why.find("quota") != std::string::npos);
    CHECK(r.st->occupancy().reserved_peaks == 0);
    CHECK(r.st->reserve(300, id, why));    // one that fits is admitted
    r.st->release_reservation(id);
    CHECK(r.st->reserve(600, id, why, /*counts_for_quota=*/false));   // a write that does not grow the store (an export copy) is not held to the quota
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
    r.start("{\"quota_gib\": 1000}");
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

// F02: an import in flight and a cold admission draw from one key reservation, in either order, and a completion never replaces a live entry.
TEST_CASE(KvflashStoragePolicyFixture, an_import_in_flight_and_a_cold_admission_never_allocate_the_same_key) {
    Gate gate;
    Rig r("kvflash_p3_unit_f02_keys", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string out = fresh_root("kvflash_p3_unit_f02_keys_out");
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 100);
    fs::copy(r.store.dir(a), out + "/a1", fs::copy_options::recursive);
    fs::copy(r.store.dir(b), out + "/b1", fs::copy_options::recursive);
    // order 1: the import is first, the cold admission (the book's new_key) of the same identity comes while its worker runs
    fs::remove_all(r.store.dir(a)); r.store.erase(a);   // gone, so its key is free again
    std::string ka;
    CHECK(r.st->import_dir(out + "/a1", false, &ka).status == 202);
    CHECK(ka == a);                                            // its own key was free again
    const std::string cold = r.store.new_key("header", "sid-a");
    CHECK(cold != ka);                                         // the reserved key is not handed out
    KvFlashSessionEntry live;
    live.key = cold; live.sid = "sid-a"; live.client = "header"; live.state = "parked-gpu"; live.identity = ID;
    r.store.upsert(live);
    // order 2: the cold admission is first (its entry is in the store), the import of the same identity comes after
    fs::remove_all(r.store.dir(b)); r.store.erase(b);   // gone, so its key is free again
    KvFlashSessionEntry cold_b;
    cold_b.key = r.store.new_key("header", "sid-b"); cold_b.sid = "sid-b"; cold_b.client = "header"; cold_b.state = "parked-gpu"; cold_b.identity = ID;
    r.store.upsert(cold_b);
    std::string kb;
    CHECK(r.st->import_dir(out + "/b1", false, &kb).status == 202);
    CHECK(kb != cold_b.key);
    gate.open();
    CHECK(w.wait_for(2));
    r.st->pump();
    CHECK(r.store.find_key(cold)->state == "parked-gpu");      // the live entries are untouched
    CHECK(r.store.find_key(cold_b.key)->state == "parked-gpu");
    REQUIRE(r.store.find_key(ka) != nullptr);
    CHECK(r.store.find_key(ka)->state == "disk");
    REQUIRE(r.store.find_key(kb) != nullptr);
    CHECK(r.store.find_key(kb)->state == "disk");
    CHECK(r.store.new_key("header", "sid-a") != ka);           // released at completion: the key is a taken entry now, not a reservation
    r.st->set_wake({});
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, an_import_completion_never_upserts_over_a_live_entry) {
    Gate gate;
    Rig r("kvflash_p3_unit_f02_live", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string out = fresh_root("kvflash_p3_unit_f02_live_out");
    const std::string a = r.add("sid-a", 100);
    fs::copy(r.store.dir(a), out + "/a1", fs::copy_options::recursive);
    std::string k;
    REQUIRE(r.st->import_dir(out + "/a1", false, &k).status == 202);
    KvFlashSessionEntry live;   // something took the key anyway (a defect elsewhere must not become data loss here)
    live.key = k; live.sid = "sid-a"; live.client = "header"; live.state = "parked-gpu"; live.identity = ID; live.last_used = 77;
    r.store.upsert(live);
    gate.open();
    CHECK(w.wait_for(1));
    r.st->pump();
    const KvFlashSessionEntry * e = r.store.find_key(k);
    REQUIRE(e != nullptr);
    CHECK(e->state == "parked-gpu");
    CHECK(e->last_used == 77);
    std::string st, err;
    REQUIRE(r.st->import_result(k, st, &err));
    CHECK(st == "failed");
    CHECK(err.find("taken") != std::string::npos);
    r.st->set_wake({});
    fs::remove_all(out);
}

// F03: shutdown quiesces the workers and the wake callback while the store and its owner are still alive.
TEST_CASE(KvflashStoragePolicyFixture, shutdown_waits_for_a_blocked_import_worker_and_reconciles_it_before_the_store_is_released) {
    Gate gate;
    Rig r("kvflash_p3_unit_f03_shutdown", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    const std::string out = fresh_root("kvflash_p3_unit_f03_shutdown_out");
    const std::string a = r.add("sid-a", 100);
    fs::copy(r.store.dir(a), out + "/a1", fs::copy_options::recursive);
    fs::remove_all(r.store.dir(a)); r.store.erase(a);   // gone, so its key is free again
    std::string k;
    REQUIRE(r.st->import_dir(out + "/a1", false, &k).status == 202);
    std::atomic<bool> returned{false};
    std::thread t([&] { r.st->shutdown(); returned.store(true); });
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    CHECK(!returned.load());                                   // the worker still holds its I/O: shutdown does not return over it
    gate.open();
    t.join();
    CHECK(returned.load());
    REQUIRE(r.store.find_key(k) != nullptr);                   // its result was committed while the store was alive
    CHECK(r.store.find_key(k)->state == "disk");
    CHECK(loads(r.store.dir(k)));
    std::string state;
    CHECK(r.st->import_result(k, state));
    CHECK(state == "imported");
    std::string k2;
    CHECK(r.st->import_dir(out + "/a1", false, &k2).status == 503);   // nothing new starts after the stop
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, clearing_the_wake_callback_waits_for_an_invocation_already_running) {
    Rig r("kvflash_p3_unit_f03_wake", "{}");
    std::mutex m;
    std::condition_variable cv;
    bool inside = false, release = false;
    r.st->set_wake([&] {
        std::unique_lock<std::mutex> lk(m);
        inside = true; cv.notify_all();
        cv.wait(lk, [&] { return release; });
    });
    std::thread ack([&] { r.st->writer_ack("k", 1, false); });   // a writer's thread: it wakes the owner
    { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&] { return inside; }); }
    std::atomic<bool> cleared{false};
    std::thread clear([&] { r.st->set_wake({}); cleared.store(true); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(!cleared.load());                                    // the callback (which may hold the server) is still running
    { std::lock_guard<std::mutex> lk(m); release = true; cv.notify_all(); }
    clear.join();
    ack.join();
    CHECK(cleared.load());
    int later = 0;
    r.st->set_wake([&] { ++later; });
    r.st->set_wake({});
    r.st->writer_ack("k", 2, false);
    CHECK(later == 0);                                         // disarmed: a late ack calls nothing
}

// F07: compaction and deletion do their file I/O on a worker; the owner only takes the claim and commits the result.
TEST_CASE(KvflashStoragePolicyFixture, a_compaction_unlinks_on_a_worker_and_the_owner_commits_it_at_the_next_pump) {
    Gate gate;
    Rig r("kvflash_p3_unit_f07_compact", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string a = r.add("sid-a", 100);
    const KvFlashResult res = r.st->compact(r.st->claim_of(a));
    CHECK(res.status == 202);                                        // dispatched, not done
    CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));   // no unlink ran on the owner thread
    CHECK(!fs::exists(r.store.dir(a) + "/compact.json"));
    CHECK(r.store.find_key(a)->state == "disk");
    CHECK(r.st->compact(r.st->claim_of(a)).status == 409);          // the claim is held while the worker runs
    CHECK(r.st->remove(r.st->claim_of(a)).status == 409);
    std::string op, state, err;
    REQUIRE(r.st->op_result(a, op, state, &err));
    CHECK((op == "compact" && state == "running"));
    gate.open();
    CHECK(w.wait_for(1));
    r.st->pump();
    CHECK(r.store.find_key(a)->state == "compact");
    CHECK(!fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(names(r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json"}));
    CHECK(r.st->op_result(a, op, state, &err));
    CHECK(state == "done");
    CHECK(r.st->claim_resume(a).status == 200);                      // released at the commit
    r.st->release(a);
    r.st->set_wake({});
}

TEST_CASE(KvflashStoragePolicyFixture, a_delete_removes_the_directory_on_a_worker_and_the_entry_goes_at_the_next_pump) {
    Gate gate;
    Rig r("kvflash_p3_unit_f07_delete", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 100);
    CHECK(r.st->remove(r.st->claim_of(a)).status == 202);
    CHECK(fs::exists(r.store.dir(a) + "/head"));                     // the recursive removal has not run on this thread
    CHECK(r.store.find_key(a) != nullptr);
    CHECK(r.st->remove(r.st->claim_of(a)).status == 409);
    gate.open();
    CHECK(w.wait_for(1));
    r.st->pump();
    CHECK(!fs::exists(r.store.dir(a)));
    CHECK(r.store.find_key(a) == nullptr);
    CHECK(r.store.find_key(b) != nullptr);
    CHECK(fs::exists(r.store.dir(b) + "/head"));
    r.st->set_wake({});
}

TEST_CASE(KvflashStoragePolicyFixture, an_export_that_deletes_the_local_session_does_it_on_a_worker_after_the_durable_copy) {
    Gate gate;
    Rig r("kvflash_p3_unit_f07_export_delete", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string out = fresh_root("kvflash_p3_unit_f07_export_delete_out");
    const std::string a = r.add("sid-a", 100);
    CHECK(r.st->export_async(r.st->claim_of(a), out).status == 202);
    gate.open();
    CHECK(w.wait_for(1));
    r.st->pump();                                                    // the copy is committed; the removal is now on a worker
    CHECK(w.wait_for(2));
    r.st->pump();
    CHECK(!fs::exists(r.store.dir(a)));
    CHECK(loads(out + "/" + a + ".g1"));
    std::string path, state;
    REQUIRE(r.st->export_result(a, path, state));
    CHECK(state == "exported");
    CHECK(r.st->claim_resume(a).status == 404);
    r.st->set_wake({});
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, the_router_answers_202_for_a_compact_or_delete_on_a_worker_and_the_session_reports_its_operation) {
    Gate gate;
    Rig r("kvflash_p3_unit_f07_router", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string a = r.add("sid-a", 100);
    CHECK(sessions_api_route(*r.st, "POST", "/v1/sessions/" + a + "/compact", "").status == 202);
    const auto g = sessions_api_route(*r.st, "GET", "/v1/sessions/" + a, "");
    CHECK(g.status == 200);
    CHECK(nlohmann::json::parse(g.body)["operation"]["state"] == "running");
    CHECK(sessions_api_route(*r.st, "DELETE", "/v1/sessions/" + a, "").status == 409);
    gate.open();
    CHECK(w.wait_for(1));
    r.st->pump();
    const auto g2 = nlohmann::json::parse(sessions_api_route(*r.st, "GET", "/v1/sessions/" + a, "").body);
    CHECK(g2["state"] == "compact");
    CHECK(g2["operation"]["state"] == "done");
    const auto d = sessions_api_route(*r.st, "DELETE", "/v1/sessions/" + a, "");
    CHECK(d.status == 202);
    CHECK(nlohmann::json::parse(d.body)["state"] == "deleting");
    CHECK(w.wait_for(2));
    r.st->pump();
    CHECK(sessions_api_route(*r.st, "GET", "/v1/sessions/" + a, "").status == 404);
    r.st->set_wake({});
}

TEST_CASE(KvflashStoragePolicyFixture, an_import_normalizes_its_metadata_before_publishing_and_a_publish_fsync_failure_leaves_nothing) {
    Rig r("kvflash_p3_unit_r14_import", "{\"quota_gib\": 1000}");
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
    REQUIRE(r.quota("{\"quota_gib\": 1000}") > 0);              // room again: the quota is not what refused the park
    sb.rewrite = false;
    CHECK(kvflash_park_admit(r.st.get(), c, e, sb, res, why));
    CHECK(res != 0);
    CHECK(r.st->occupancy().reserved_peaks >= kvflash_commit_bytes(c));
    CHECK(kvflash_park_admit(nullptr, c, e, sb, res, why));  // no storage: nothing to ask
}

// R03 (P3 part): the engine wires book.resume_claim to the storage claim; a denial is busy (the request retries), never a cold admit
TEST_CASE(KvflashStoragePolicyFixture, a_resume_denied_by_a_storage_action_is_busy_and_a_granted_one_holds_the_claim_for_the_restore) {
    Rig r("kvflash_p3_unit_r03", "{}");
    const std::string a = r.add("D", 100);
    KvFlashSessionBook book;
    book.init(2);
    book.enable_store(r.store, ID);
    kvflash_attach_storage(book, r.st.get());   // the engine's wiring, not a hook this test installs itself
    std::vector<int32_t> prompt = book.store().find_key(a)->history;
    prompt.push_back(9);
    const std::vector<bool> active = {false, false};
    REQUIRE(r.st->take(r.st->claim_of(a)).ok());                   // an export (or compaction) holds the key
    CHECK(book.plan_restore("header", "D", prompt, 0, active).pick == KvFlashRestorePick::busy);   // not a cold admission
    r.st->release(a);
    const auto plan = book.plan_restore("header", "D", prompt, 0, active);
    CHECK(plan.pick == KvFlashRestorePick::load);
    CHECK(r.st->remove(r.st->claim_of(a)).status == 409);          // the restore holds the claim until the engine releases it
    r.st->release(a);
    CHECK(!kvflash_resume_claim(nullptr));                         // no storage: nothing is wired, the book never refuses
    kvflash_attach_storage(book, nullptr);
    CHECK(!book.resume_claim);
    REQUIRE(r.st->take(r.st->claim_of(a)).ok());
    CHECK(book.plan_restore("header", "D", prompt, 0, active).pick == KvFlashRestorePick::load);   // unwired: the claim is not asked
    r.st->release(a);
}

// Re-review round 2.

// F04: the export deletes the source only after the entry it published is the directory it verified
TEST_CASE(KvflashStoragePolicyFixture, an_export_whose_staging_directory_is_swapped_after_verification_fails_and_keeps_the_source) {
    Rig r("kvflash_p3_unit_r2_swap");
    const std::string out = fresh_root("kvflash_p3_unit_r2_swap_out");
    std::atomic<int> calls{0};
    const std::string a = r.add("sid-a", 100);
    r.env.fsync_dir = [&, a](const std::string & d) {
        if (calls.fetch_add(1) == 0) {   // the staging directory's own fsync: verified, not yet renamed
            fs::rename(out + "/" + a + ".g1.tmp", out + "/moved");
            fs::create_directory(out + "/" + a + ".g1.tmp");
            write_text(out + "/" + a + ".g1.tmp/theirs", "not the verified copy");
        }
        return kvflash_fsync_dir(d);
    };
    r.start("{}");   // after_export = delete
    const KvFlashResult res = r.st->export_session(r.st->claim_of(a), out);
    CHECK(!res.ok());
    CHECK(loads(r.store.dir(a)));
    CHECK(r.store.find_key(a) != nullptr);
    CHECK(slurp(out + "/" + a + ".g1/theirs") == "not the verified copy");   // what is published is left alone
    std::string path, state;
    REQUIRE(r.st->export_result(a, path, state));
    CHECK(state == "failed");
    fs::remove_all(out);
}

// F04: only a name this export created with O_EXCL is ever unlinked on withdrawal
TEST_CASE(KvflashStoragePolicyFixture, a_file_that_was_in_the_staging_directory_before_the_export_created_it_survives_the_withdrawal) {
    Rig r("kvflash_p3_unit_r2_excl");
    const std::string out = fresh_root("kvflash_p3_unit_r2_excl_out");
    const std::string a = r.add("sid-a", 100);
    KvFlashExportPlan plan;
    REQUIRE(r.st->export_begin(r.st->claim_of(a), out, plan).ok());
    const std::string fifo = r.store.dir(a) + "/gate";
    REQUIRE(::mkfifo(fifo.c_str(), 0644) == 0);
    plan.files = {"head", "gate"};   // the copy of "gate" blocks on the fifo until the test lets it go
    KvFlashResult res;
    std::thread t([&] { res = r.st->export_copy(plan); });
    for (int i = 0; i < 2000 && !fs::exists(out + "/" + a + ".g1.tmp/head"); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    write_text(out + "/" + a + ".g1.tmp/gate", "theirs");   // appears in the staging directory after the export made it
    const int w = ::open(fifo.c_str(), O_WRONLY);
    if (w >= 0) ::close(w);
    t.join();
    CHECK(!res.ok());
    CHECK(slurp(out + "/" + a + ".g1.tmp/gate") == "theirs");
    CHECK(!fs::exists(out + "/" + a + ".g1.tmp/head"));   // what it did make is withdrawn
    r.st->release(a);
    fs::remove_all(out);
}

// F02/F07: a refused dispatch gives back what was reserved for it
TEST_CASE(KvflashStoragePolicyFixture, an_import_the_storage_refuses_or_cannot_start_leaves_no_key_and_no_reserved_bytes) {
    for (const bool stopped : {true, false}) {
        Rig r("kvflash_p3_unit_r2_import_refused", "{}");
        r.env.async_export = true;
        r.env.spawn_hook = [] { throw std::system_error(EAGAIN, std::generic_category(), "thread"); };
        r.start("{\"quota_gib\": 1000}");
        const std::string out = fresh_root("kvflash_p3_unit_r2_import_refused_out");
        const std::string a = r.add("sid-a", 100);
        fs::copy(r.store.dir(a), out + "/a1", fs::copy_options::recursive);
        const uint64_t reserved = r.st->occupancy().reserved_peaks;
        const auto keys = r.store.keys();
        if (stopped) r.st->shutdown();
        std::string k;
        const KvFlashResult res = r.st->import_dir(out + "/a1", false, &k);
        CHECK(!res.ok());
        CHECK(res.status == 503);
        CHECK(r.st->occupancy().reserved_peaks == reserved);
        CHECK(r.store.keys() == keys);
        std::string state;
        CHECK(!r.st->import_result(k, state));
        fs::remove_all(out);
    }
}

TEST_CASE(KvflashStoragePolicyFixture, a_compaction_or_delete_that_cannot_be_dispatched_is_failed_and_its_claim_is_released) {
    for (const bool del : {false, true}) {
        Rig r("kvflash_p3_unit_r2_dispatch", "{}");
        r.env.async_export = true;
        r.env.spawn_hook = [] { throw std::system_error(EAGAIN, std::generic_category(), "thread"); };
        r.start("{\"quota_gib\": 1000}");
        const std::string a = r.add("sid-a", 100);
        const KvFlashResult res = del ? r.st->remove(r.st->claim_of(a)) : r.st->compact(r.st->claim_of(a));
        CHECK(!res.ok());
        std::string op, state, err;
        REQUIRE(r.st->op_result(a, op, state, &err));
        CHECK(op == (del ? "delete" : "compact"));
        CHECK(state == "failed");
        CHECK(!err.empty());
        CHECK(r.store.find_key(a)->state == "disk");
        CHECK(loads(r.store.dir(a)));
        CHECK(r.st->claim_resume(a).status == 200);   // not left running with the key claimed
        r.st->release(a);
    }
}

// F07: the steward's unlink retry and the delete's directory fsync run on a worker
TEST_CASE(KvflashStoragePolicyFixture, the_unlink_retry_runs_on_a_worker_and_the_owner_commits_it) {
    Rig r("kvflash_p3_unit_r2_retry", "{}");
    r.env.async_export = true;
    std::mutex m;
    std::vector<std::thread::id> unlinkers;
    const auto base = r.env.unlink_file;
    r.env.unlink_file = [&](const std::string & p) { { std::lock_guard<std::mutex> lk(m); unlinkers.push_back(std::this_thread::get_id()); } return base(p); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string a = r.add("sid-a", 100);
    r.unlink_fails = {kvflash_bundle_log_name(1)};
    REQUIRE(r.st->compact(r.st->claim_of(a)).status == 202);
    REQUIRE(w.wait_for(1));
    r.st->pump();                                                            // commits the compaction; its pass starts a retry that fails again
    REQUIRE(w.wait_for(2));
    r.st->pump();
    CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));   // the unlink is owed
    r.unlink_fails.clear();
    { std::lock_guard<std::mutex> lk(m); unlinkers.clear(); }
    r.st->tick();
    REQUIRE(w.wait_for(3));
    r.st->pump();
    CHECK(!fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(r.ent(a).bytes_live == tree_bytes(r.store.dir(a)));
    {
        std::lock_guard<std::mutex> lk(m);
        CHECK(!unlinkers.empty());
        for (const auto & id : unlinkers) CHECK(id != std::this_thread::get_id());   // never on the owner
    }
    r.st->set_wake({});
}

TEST_CASE(KvflashStoragePolicyFixture, a_delete_fsyncs_its_parent_directory_on_a_worker_and_a_failed_fsync_is_a_failed_operation_the_session_reports) {
    Rig r("kvflash_p3_unit_r2_delete_fsync", "{}");
    r.env.async_export = true;
    std::mutex m;
    std::vector<std::thread::id> syncers;
    std::atomic<bool> fail{false};
    r.env.fsync_dir = [&](const std::string & d) { { std::lock_guard<std::mutex> lk(m); syncers.push_back(std::this_thread::get_id()); } return !fail.load() && kvflash_fsync_dir(d); };
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string a = r.add("sid-a", 100), b = r.add("sid-b", 100);
    { std::lock_guard<std::mutex> lk(m); syncers.clear(); }
    REQUIRE(r.st->remove(r.st->claim_of(a)).status == 202);
    REQUIRE(w.wait_for(1));
    r.st->pump();
    {
        std::lock_guard<std::mutex> lk(m);
        CHECK(!syncers.empty());
        for (const auto & id : syncers) CHECK(id != std::this_thread::get_id());
    }
    fail.store(true);
    REQUIRE(r.st->remove(r.st->claim_of(b)).status == 202);
    REQUIRE(w.wait_for(2));
    r.st->pump();
    CHECK(r.store.find_key(b) == nullptr);            // the directory is gone, the entry with it
    std::string op, state, err;
    REQUIRE(r.st->op_result(b, op, state, &err));
    CHECK((op == "delete" && state == "failed"));
    const auto g = sessions_api_route(*r.st, "GET", "/v1/sessions/" + b, "");   // not a 404 that says it never existed
    CHECK(g.status == 200);
    CHECK(nlohmann::json::parse(g.body)["operation"]["state"] == "failed");
    CHECK(sessions_api_route(*r.st, "GET", "/v1/sessions/" + a, "").status == 404);   // a delete that went through stays gone
    r.st->set_wake({});
}

// F07: the 202 for an import names the key it reserved, and the key answers GET at once
TEST_CASE(KvflashStoragePolicyFixture, the_202_for_an_import_names_the_reserved_key_and_that_key_is_importing_before_the_worker_runs) {
    Gate gate;
    Rig r("kvflash_p3_unit_r2_import_key", "{}");
    OpenOnExit guard{gate};
    r.env.async_export = true;
    r.env.worker_gate = [&gate] { gate.wait(); };
    r.start("{\"quota_gib\": 1000}");
    const std::string out = fresh_root("kvflash_p3_unit_r2_import_key_out");
    const std::string a = r.add("sid-a", 100);
    fs::copy(r.store.dir(a), out + "/a1", fs::copy_options::recursive);
    const auto res = sessions_api_route(*r.st, "POST", "/v1/sessions/import", "{\"path\":\"" + out + "/a1\"}");
    REQUIRE(res.status == 202);
    const std::string k = nlohmann::json::parse(res.body)["key"];
    CHECK(!k.empty());
    CHECK(k != a);
    const auto g = sessions_api_route(*r.st, "GET", "/v1/sessions/" + k, "");
    CHECK(g.status == 200);
    CHECK(nlohmann::json::parse(g.body)["state"] == "importing");
    gate.open();
    r.st->shutdown();
    fs::remove_all(out);
}

// Round 3: a failing unlink must not keep the claim of a session the policy wants to export
TEST_CASE(KvflashStoragePolicyFixture, an_unlink_that_keeps_failing_never_keeps_the_policy_from_exporting_the_session) {
    Rig r("kvflash_p3_unit_r3_starve", "{}");
    r.env.async_export = true;
    r.start("{\"quota_gib\": 1000}");
    Wakes w;
    r.st->set_wake([&] { w.n.fetch_add(1); });
    const std::string out = fresh_root("kvflash_p3_unit_r3_starve_out");
    std::string err;
    MUST(r.st->set_policy("{\"quota_gib\": 1000, \"action\": \"export\", \"after_export\": \"keep\", \"max_age_days\": 1, \"export_path\": \"" + out + "\"}", err));
    const std::string a = r.add("sid-a", r.now - 3 * 86400);
    r.unlink_fails = {kvflash_bundle_log_name(1)};
    REQUIRE(r.st->compact(r.st->claim_of(a)).status == 202);
    REQUIRE(w.wait_for(1));
    const auto work = r.st->tick();   // the commit releases the compaction's claim; the unlink is still owed
    REQUIRE(work.size() == 1);        // and the policy gets the session before a retry can claim it
    CHECK(work[0].key == a);
    r.st->set_wake({});
    fs::remove_all(out);
}

// ── the auto reserve (eviction stage 0, P1) ────────────────────────────────────────────────
// It was max(10% of the disk, 20 GiB): 204 GiB on a 2 TB drive, so every park was refused on a box with 28 GB free.

TEST_CASE(KvflashStoragePolicyFixture, the_auto_reserve_is_ten_percent_of_the_disk_capped_at_4_gib) {
    constexpr uint64_t G = 1ull << 30;
    CHECK(kvflash_auto_reserve_bytes(1900 * G, G) == 4 * G);      // the 1.9 TB WSL drive: 4 GiB, not 204
    CHECK(kvflash_auto_reserve_bytes(2000 * G, G) == 4 * G);
    CHECK(kvflash_auto_reserve_bytes(40 * G, G) == 4 * G);        // 10% = 4 GiB exactly
    CHECK(kvflash_auto_reserve_bytes(20 * G, G) == 2 * G);        // a small disk keeps 10%
    CHECK(kvflash_auto_reserve_bytes(0, G) == 0);
}

// ── stage 3: a park refused for disk room asks the steward for room (the policy's action, oldest disk session first) ──

namespace {
// A rig whose free space is CAP minus the store, with a parked owner whose park is `shortfall` bytes short of fitting.
struct RoomRig {
    Rig r;
    std::string owner;
    KvFlashBundleCommit c;
    KvFlashSessionEntry e;
    KvFlashSlotBundle sb;
    uint64_t cap = 1ull << 40;
    RoomRig(const char * name, const std::string & policy, const std::function<void(Rig &)> & pre = {}) : r(name) {
        r.env.free_bytes = [this] { return cap > tree_bytes(r.root) ? cap - tree_bytes(r.root) : 0; };
        if (pre) pre(r);
        r.start(policy.c_str());
        owner = r.add("sid-owner", 100, 6, "parked-gpu");
        c = make_commit(r.store.dir(owner), 6, 0x60);
        sb = KvFlashSlotBundle::adopted(owner, "sid-owner", "header", 90, KvFlashSlotHead{});
        sb.head.gen = 1;
    }
    // call once the sessions are added: the cap that leaves the park one byte short of the free space it needs
    void squeeze() {
        e = r.ent(owner);
        uint64_t id = 0;
        std::string why;
        MUST(r.st->reserve_commit(c, e, id, why));
        const uint64_t peak = r.st->occupancy().reserved_peaks;
        r.st->release_reservation(id);
        cap = tree_bytes(r.root) + r.st->reserve_bytes() + peak - 1;
        MUST(!r.st->reserve_commit(c, e, id, why));
    }
    bool admit(uint64_t & res, std::string & why) { return kvflash_park_admit(r.st.get(), c, e, sb, res, why); }
};
}  // namespace

TEST_CASE(KvflashStoragePolicyFixture, stage3_delete_oldest_removes_the_oldest_disk_session_and_the_park_proceeds) {
    RoomRig g("kvflash_s3_del", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"delete-oldest\"}");
    const std::string old = g.r.add("sid-old", 200, 20), mid = g.r.add("sid-mid", 300, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(g.admit(res, why));
    CHECK(res != 0);
    CHECK(!g.sb.rewrite);
    CHECK(g.r.store.find_key(old) == nullptr);
    CHECK(g.r.store.find_key(mid) != nullptr);
    CHECK(g.r.ent(g.owner).state == "parked-gpu");
    CHECK(fs::exists(g.r.store.dir(g.owner) + "/head"));
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_compact_compacts_the_oldest_disk_session_and_the_park_proceeds) {
    RoomRig g("kvflash_s3_cmp", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"compact\"}");
    const std::string old = g.r.add("sid-old", 200, 20), mid = g.r.add("sid-mid", 300, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(g.admit(res, why));
    CHECK(!g.sb.rewrite);
    CHECK(g.r.store.find_key(old) != nullptr);
    CHECK(g.r.ent(old).state == "compact");
    CHECK(g.r.store.find_key(mid) != nullptr);   // compacted only if the oldest alone did not free enough, never deleted
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_export_moves_the_oldest_disk_session_out_and_the_park_proceeds) {
    const std::string out = fresh_root("kvflash_s3_exp_out");
    RoomRig g("kvflash_s3_exp", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"export\", \"export_path\":\"" + out + "\", \"after_export\":\"delete\"}");
    const std::string old = g.r.add("sid-old", 200, 20), mid = g.r.add("sid-mid", 300, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(g.admit(res, why));
    CHECK(!g.sb.rewrite);
    CHECK(g.r.store.find_key(old) == nullptr);
    CHECK(fs::exists(out + "/" + old + ".g1"));
    CHECK(g.r.store.find_key(mid) != nullptr);
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_ask_queues_one_space_question_names_it_and_refuses_this_park) {
    RoomRig g("kvflash_s3_ask", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"ask\"}");
    const std::string old = g.r.add("sid-old", 200, 20), mid = g.r.add("sid-mid", 300, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));
    CHECK(g.sb.rewrite);
    CHECK(res == 0);
    const auto p = g.r.st->pending();
    REQUIRE(p.size() == 1);
    CHECK(p[0].reason == "space");
    CHECK(p[0].key == old);                          // the oldest disk session
    CHECK(why.find(p[0].id) != std::string::npos);   // the log line names the question
    g.sb.rewrite = false;
    CHECK(!g.admit(res, why));                       // the same space event again: still one question
    CHECK(g.r.st->pending().size() == 1);
    CHECK(g.r.store.find_key(old) != nullptr);
    CHECK(g.r.store.find_key(mid) != nullptr);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_parked_held_or_busy_session_is_never_touched_to_make_room) {
    for (const char * action : {"delete-oldest", "compact"}) {
        RoomRig g("kvflash_s3_prot", std::string("{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"") + action + "\"}");
        const std::string parked = g.r.add("sid-parked", 150, 20, "parked-gpu"), busy = g.r.add("sid-busy", 160, 20), held = g.r.add("sid-held", 170, 20);
        g.squeeze();
        REQUIRE(g.r.st->take(g.r.st->claim_of(busy)).ok());
        g.r.st->writer_begin(held);
        uint64_t res = 0;
        std::string why;
        CHECK(!g.admit(res, why));                   // nothing eligible: refused as before
        CHECK(g.sb.rewrite);
        for (const std::string & k : {parked, busy, held}) {
            REQUIRE(g.r.store.find_key(k) != nullptr);
            CHECK(g.r.ent(k).state != "compact");
            CHECK(fs::exists(g.r.store.dir(k) + "/head"));
        }
        g.r.st->release(busy);
    }
}

// ── stage 3 review fixes ────────────────────────────────────────────────────────────────

// Coordinator ruling on P1: only the live session's immediate archive is exempt; older generations follow the policy
TEST_CASE(KvflashStoragePolicyFixture, stage3_only_the_immediate_archive_of_a_live_session_is_exempt_older_generations_are_candidates) {
    RoomRig g("kvflash_s3r_chain", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"delete-oldest\"}");
    const std::string a = g.r.add("sid-a", 200, 20), b = g.r.add("sid-b", 300, 20), c = g.r.add("sid-c", 400, 20, "parked-gpu");
    g.r.ent(a).archive = b; g.r.ent(b).archive_of = a;   // A -> B -> C, C live in a slot
    g.r.ent(b).archive = c; g.r.ent(c).archive_of = b;
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(g.admit(res, why));
    CHECK(g.r.store.find_key(a) == nullptr);   // the grandparent follows the policy
    CHECK(g.r.store.find_key(b) != nullptr);   // the immediate archive does not
    CHECK(g.r.ent(c).state == "parked-gpu");
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_cascade_export_is_committed_before_the_park_retries_even_if_the_copy_finishes_inside_the_commit_gap) {
    const std::string out = fresh_root("kvflash_s3r_gap_out");
    std::atomic<int> gaps{0};
    RoomRig g("kvflash_s3r_gap", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"export\", \"export_path\":\"" + out + "\", \"after_export\":\"delete\"}", [&](Rig & r) {
        r.env.async_export = true;
        r.env.commit_gap = [&gaps] { if (gaps.fetch_add(1) == 0) std::this_thread::sleep_for(std::chrono::milliseconds(300)); };   // the worker finishes in here
    });
    const std::string old = g.r.add("sid-old", 200, 20), mid = g.r.add("sid-mid", 300, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(g.admit(res, why));
    CHECK(g.r.store.find_key(old) == nullptr);        // the local delete (the export's continuation) ran before the retry
    CHECK(g.r.store.find_key(mid) != nullptr);        // and one export was enough
    CHECK(names(out) == (std::set<std::string>{old + ".g1"}));
    g.r.st.reset();
    fs::remove_all(out);
}

namespace {
struct TimedGate {   // the first `n` workers wait here until released or `ms` pass (a test must never hang on a bug)
    std::mutex m; std::condition_variable cv; bool open_ = false; int seen = 0, n; int64_t ms;
    TimedGate(int n_, int64_t ms_) : n(n_), ms(ms_) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        if (seen++ >= n) return;
        cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return open_; });
    }
    void open() { { std::lock_guard<std::mutex> lk(m); open_ = true; } cv.notify_all(); }
};
double ms_since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }
}  // namespace

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_park_does_not_wait_for_a_worker_the_cascade_did_not_start) {
    const std::string out = fresh_root("kvflash_s3r_stall_out");
    TimedGate gate(1, 1500);
    RoomRig g("kvflash_s3r_stall", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"delete-oldest\"}", [&](Rig & r) {
        r.env.async_export = true;
        r.env.worker_gate = [&gate] { gate.wait(); };
    });
    const std::string unrelated = g.r.add("sid-x", 150, 20), old = g.r.add("sid-old", 200, 20), mid = g.r.add("sid-mid", 300, 20);
    REQUIRE(g.r.st->export_async(g.r.st->claim_of(unrelated), out).ok());   // the first worker: stuck at the gate, not the cascade's
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(g.admit(res, why));
    CHECK(ms_since(t0) < 600);
    CHECK(g.r.store.find_key(old) == nullptr);
    gate.open();
    g.r.st.reset();
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_when_the_cascade_action_outlasts_the_bound_the_park_is_refused_and_the_next_park_finds_room) {
    TimedGate gate(1, 1500);
    RoomRig g("kvflash_s3r_bound", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"delete-oldest\"}", [&](Rig & r) {
        r.env.async_export = true;
        r.env.room_wait_ms = 100;
        r.env.worker_gate = [&gate] { gate.wait(); };
    });
    const std::string old = g.r.add("sid-old", 200, 20), mid = g.r.add("sid-mid", 300, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(!g.admit(res, why));                       // the delete is stuck: this park is refused, within the bound
    CHECK(ms_since(t0) < 900);
    CHECK(g.sb.rewrite);
    CHECK(why.find("still running") != std::string::npos);
    gate.open();
    for (int i = 0; i < 400 && g.r.store.find_key(old); ++i) { g.r.st->pump(); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    CHECK(g.r.store.find_key(old) == nullptr);       // the action finished in the background
    g.sb.rewrite = false;
    CHECK(g.admit(res, why));                        // so the next park finds room
    CHECK(g.r.store.find_key(mid) != nullptr);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_ask_reuses_the_open_age_question_of_its_candidate_instead_of_adding_a_space_one) {
    RoomRig g("kvflash_s3r_dedup", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"ask\", \"max_age_days\": 1}");
    const std::string old = g.r.add("sid-old", g.r.now - 90 * 86400, 20), mid = g.r.add("sid-mid", g.r.now - 10, 20);
    g.r.st->tick();                                  // the tick's age question about `old`
    REQUIRE(g.r.st->pending().size() == 1);
    const std::string id = g.r.st->pending()[0].id;
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));
    REQUIRE(g.r.st->pending().size() == 1);          // no second question for the same session
    CHECK(g.r.st->pending()[0].id == id);
    CHECK(why.find(id) != std::string::npos);        // and the refusal names it
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_at_most_one_space_question_is_open_across_the_cascade_and_the_dead_log_pass) {
    Rig r("kvflash_s3r_one_space", "{\"reserve_gib\": 1, \"quota_gib\": 1000, \"log_dead_mib\": 1, \"action\":\"ask\"}");
    const std::string a = r.add("sid-a", r.now - 1000, 40, "disk", 10), b = r.add("sid-b", r.now - 900, 40, "disk", 10);
    r.st->tick();
    r.ent(a).bytes_dead = 30 * RB; r.ent(b).bytes_dead = 30 * RB;
    r.free_b = GIB + 10 * RB + kKvFlashLogHeader - 1;
    r.st->tick();
    int space = 0;
    for (const auto & p : r.st->pending()) space += p.reason == "space";
    CHECK(space == 1);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_refused_cascade_action_is_named_in_the_parks_refusal) {
    RoomRig g("kvflash_s3r_silent", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"export\", \"export_path\":\"/nonexistent-kvflash-s3r\"}");
    const std::string old = g.r.add("sid-old", 200, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));
    CHECK(why.find("does not exist") != std::string::npos);   // the last action's refusal, not only the original reason
    CHECK(why.find(old) != std::string::npos);
}

// the export publication on a filesystem without RENAME_NOREPLACE (drvfs on WSL: EINVAL)
namespace {
const auto kNoNoreplace = [](int, const char *, const char *) { errno = EINVAL; return -1; };
}
TEST_CASE(KvflashStoragePolicyFixture, stage3_an_export_without_rename_noreplace_falls_back_to_an_exclusive_directory_with_head_last_and_deletes_the_source_after_it_verifies) {
    const std::string out = fresh_root("kvflash_s3r_fb_out");
    Rig r("kvflash_s3r_fb", "{}");
    r.env.rename_noreplace = kNoNoreplace;
    r.start("{\"quota_gib\": 1000}");
    const std::string a = r.add("sid-a", 100);
    std::string err;
    REQUIRE(r.st->set_policy(("{\"export_path\":\"" + out + "\", \"after_export\":\"delete\"}").c_str(), err));
    const auto res = r.st->export_session(r.st->claim_of(a), "");
    REQUIRE(res.status == 200);
    CHECK(names(out + "/" + a + ".g1") == (std::set<std::string>{"head", "meta.json", kvflash_bundle_log_name(1), kvflash_bundle_slab_name(2)}));
    CHECK(names(out) == (std::set<std::string>{a + ".g1"}));   // no staging directory left
    CHECK(loads(out + "/" + a + ".g1"));
    CHECK(!fs::exists(r.store.dir(a)));
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_the_fallback_keeps_the_source_and_withdraws_the_copy_when_its_directory_fsync_fails_and_keeps_it_with_after_export_keep) {
    const std::string out = fresh_root("kvflash_s3r_fb2_out");
    bool fail_parent = false;
    int fsyncs = 0;   // in place: the copy's directory, then its parent (the second)
    Rig r("kvflash_s3r_fb2", "{}");
    r.env.rename_noreplace = kNoNoreplace;
    r.env.fsync_dir = [&](const std::string &) { return !(fail_parent && ++fsyncs >= 2); };
    r.start("{\"quota_gib\": 1000}");
    const std::string a = r.add("sid-a", 100);
    std::string err;
    REQUIRE(r.st->set_policy(("{\"export_path\":\"" + out + "\", \"after_export\":\"delete\"}").c_str(), err));
    fail_parent = true;
    CHECK(!r.st->export_session(r.st->claim_of(a), "").ok());
    CHECK(r.store.find_key(a) != nullptr);             // the source stays
    CHECK(loads(r.store.dir(a)));
    CHECK(names(out).empty());                         // nothing published
    fail_parent = false;
    REQUIRE(r.st->set_policy(("{\"export_path\":\"" + out + "\", \"after_export\":\"keep\"}").c_str(), err));
    CHECK(r.st->export_session(r.st->claim_of(a), "").ok());
    CHECK(r.store.find_key(a) != nullptr);             // keep: the source stays after a verified copy
    CHECK(loads(out + "/" + a + ".g1"));
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_fallback_directory_interrupted_before_head_is_not_an_export_and_import_refuses_it) {
    const std::string out = fresh_root("kvflash_s3r_fb3_out");
    Rig r("kvflash_s3r_fb3", "{\"quota_gib\": 1000}");
    const std::string a = r.add("sid-a", 100);
    fs::create_directories(out + "/torn");
    for (const char * f : {"meta.json", "chunks.1.log", "slab.2"}) if (fs::exists(r.store.dir(a) + "/" + f)) fs::copy_file(r.store.dir(a) + "/" + f, out + "/torn/" + f);   // everything but head
    const size_t n = r.store.entries().size();
    std::string key;
    CHECK(!r.st->import_dir(out + "/torn", false, &key).ok());
    CHECK(r.store.entries().size() == n);
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_destination_without_rename_noreplace_is_found_by_a_probe_and_the_copy_is_made_once_in_place) {
    const std::string out = fresh_root("kvflash_s3r_probe_out");
    std::vector<std::string> froms;
    Rig r("kvflash_s3r_probe", "{}");
    r.env.rename_noreplace = [&](int, const char * from, const char *) { froms.push_back(from); errno = EINVAL; return -1; };
    r.start("{\"quota_gib\": 1000}");
    const std::string a = r.add("sid-a", 100);
    std::string err;
    REQUIRE(r.st->set_policy(("{\"export_path\":\"" + out + "\", \"after_export\":\"delete\"}").c_str(), err));
    REQUIRE(r.st->export_session(r.st->claim_of(a), "").status == 200);
    REQUIRE(froms.size() == 1);
    CHECK(froms[0].find(".np0") != std::string::npos);   // the probe of an empty directory, not the publication of a staged copy
    CHECK(loads(out + "/" + a + ".g1"));
    CHECK(names(out) == (std::set<std::string>{a + ".g1"}));      // the probe left nothing
    fs::remove_all(out);
}

// ── stage 3 re-review ───────────────────────────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, stage3_the_whole_cascade_shares_one_bound_so_several_slow_candidates_do_not_stack_their_waits) {
    TimedGate gate(100, 250);   // every worker takes 250 ms
    RoomRig g("kvflash_s3rr_deadline", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"delete-oldest\"}", [&](Rig & r) {
        r.env.async_export = true;
        r.env.room_wait_ms = 400;
        r.env.worker_gate = [&gate] { gate.wait(); };
    });
    for (int i = 0; i < 4; ++i) g.r.add("sid-" + std::to_string(i), 200 + i, 20);
    g.squeeze();
    g.cap -= 2 * tree_bytes(g.r.store.dir(g.r.store.entries()[1].key));   // three deletions are needed, not one
    uint64_t res = 0;
    std::string why;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(!g.admit(res, why));
    CHECK(ms_since(t0) < 600);   // one bound (400 ms) plus a worker, not 3 x 250 ms stacked
    CHECK(g.sb.rewrite);
    gate.open();
    g.r.st.reset();
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_the_cascade_does_not_export_when_the_local_copy_is_kept_because_that_frees_nothing) {
    const std::string out = fresh_root("kvflash_s3rr_keep_out");
    RoomRig g("kvflash_s3rr_keep", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"export\", \"export_path\":\"" + out + "\", \"after_export\":\"keep\"}");
    const std::string old = g.r.add("sid-old", 200, 20);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));
    CHECK(names(out).empty());                         // nothing was copied
    CHECK(g.r.store.find_key(old) != nullptr);
    CHECK(why.find("after_export") != std::string::npos);
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_key_exported_again_after_it_came_back_lands_beside_the_first_export_and_both_import) {
    const std::string out = fresh_root("kvflash_s3rr_again_out");
    Rig r("kvflash_s3rr_again", "{\"quota_gib\": 1000}");
    std::string err;
    REQUIRE(r.st->set_policy(("{\"export_path\":\"" + out + "\", \"after_export\":\"delete\"}").c_str(), err));
    const std::string a = r.add("sid-a", 100);
    REQUIRE(r.st->export_session(r.st->claim_of(a), "").status == 200);
    REQUIRE(r.store.find_key(a) == nullptr);
    const std::string b = r.add("sid-a", 200);       // the same session comes back to disk: same key, generations start over
    REQUIRE(b == a);
    const auto second = r.st->export_session(r.st->claim_of(b), "");
    CHECK(second.status == 200);
    const auto n = names(out);
    CHECK(n.size() == 2);                            // neither overwrote the other
    for (const auto & d : n) {
        CHECK(d.rfind(a + ".g", 0) == 0);
        std::string key;
        CHECK(r.st->import_dir(out + "/" + d, false, &key).status == 200);   // a name that is not the key still imports
    }
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_an_in_place_export_whose_directory_is_swapped_before_success_fails_and_keeps_the_source) {
    Rig r("kvflash_s3rr_inplace_swap");
    const std::string out = fresh_root("kvflash_s3rr_inplace_swap_out");
    r.env.rename_noreplace = kNoNoreplace;
    std::atomic<int> calls{0};
    const std::string a = r.add("sid-a", 100);
    std::string dst;
    r.env.fsync_dir = [&](const std::string & d) {
        if (calls.fetch_add(1) == 0) {   // the copy's own directory: verified, not yet declared a success
            for (const auto & de : fs::directory_iterator(out)) dst = de.path().string();
            fs::rename(dst, out + "/moved");
            fs::create_directory(dst);
            write_text(dst + "/theirs", "not the verified copy");
        }
        return kvflash_fsync_dir(d);
    };
    r.start("{}");
    const KvFlashResult res = r.st->export_session(r.st->claim_of(a), out);
    CHECK(!res.ok());
    CHECK(r.store.find_key(a) != nullptr);
    CHECK(loads(r.store.dir(a)));
    CHECK(slurp(dst + "/theirs") == "not the verified copy");
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_a_failure_of_an_accepted_cascade_worker_is_named_in_the_parks_refusal) {
    const std::string out = fresh_root("kvflash_s3rr_workerfail_out");
    std::string victim_log;
    RoomRig g("kvflash_s3rr_workerfail", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"export\", \"export_path\":\"" + out + "\"}", [&](Rig & r) {
        r.env.async_export = true;
        r.env.worker_gate = [&] { ::chmod(victim_log.c_str(), 0); };   // the worker cannot read the log; same bytes, nothing freed
    });
    const std::string old = g.r.add("sid-old", 200, 20);
    victim_log = g.r.store.dir(old) + "/" + kvflash_bundle_log_name(1);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));
    CHECK(why.find("cannot read") != std::string::npos);   // the worker's error, not only the start's
    fs::remove_all(out);
}

// ── round 3 (Codex re-review of 70fd149) ──────────────────────────────────────────────────

TEST_CASE(KvflashStoragePolicyFixture, stage3_no_completion_callback_starts_once_the_cascade_deadline_has_passed) {
    std::atomic<int> started{0};
    RoomRig g("kvflash_s3r3_deadline", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"delete-oldest\"}", [&](Rig & r) {
        r.env.async_export = true;
        r.env.room_wait_ms = 100;
        r.env.before_commit = [&started] { started.fetch_add(1); std::this_thread::sleep_for(std::chrono::milliseconds(200)); };   // each owner-side commit (index write, fsync) is slow
    });
    const std::string old = g.r.add("sid-old", 200, 20);
    std::vector<std::string> others;
    for (int i = 0; i < 3; ++i) others.push_back(g.r.add("sid-u" + std::to_string(i), 300 + i, 20));
    for (const auto & k : others) REQUIRE(g.r.st->remove(g.r.st->claim_of(k)).status == 202);   // three unrelated deletes: the directories go at once, their commits wait for the owner
    for (int i = 0; i < 2000 && g.r.st->finished_pending() < 3; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(g.r.st->finished_pending() == 3);   // the queue is ready: three finished workers, none committed
    g.squeeze();   // now the park is one byte short, and only `old` can make the room
    uint64_t res = 0;
    std::string why;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(!g.admit(res, why));
    CHECK(ms_since(t0) < 450);          // the bound: one slow commit, then no callback starts after the deadline (before: all of them, 800 ms)
    CHECK(started.load() == 1);         // exactly one commit started before the deadline
    for (int i = 0; i < 400 && g.r.store.find_key(old); ++i) { g.r.st->pump(); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    CHECK(g.r.store.find_key(old) == nullptr);   // the cascade's own delete, and what was left over, are committed by the next pumps
    for (int i = 0; i < 100; ++i) { g.r.st->pump(); std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
    CHECK(started.load() >= 4);                  // all of them, in the end
    g.sb.rewrite = false;
    CHECK(g.admit(res, why));
}

TEST_CASE(KvflashStoragePolicyFixture, stage3_an_export_whose_local_removal_failed_is_named_in_the_parks_refusal) {
    const std::string out = fresh_root("kvflash_s3r3_rmfail_out");
    std::string victim;
    RoomRig g("kvflash_s3r3_rmfail", "{\"reserve_gib\": 1, \"quota_gib\": 0, \"action\":\"export\", \"after_export\":\"delete\", \"export_path\":\"" + out + "\"}", [&](Rig & r) {
        r.env.async_export = true;
        r.env.worker_gate = [&victim] { ::chmod(victim.c_str(), 0555); };   // the copy only reads; the removal that follows cannot unlink one file
    });
    const std::string old = g.r.add("sid-old", 200, 20);
    victim = g.r.store.dir(old);
    g.squeeze();
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));   // the copy is durable but the session stays: no room was made
    CHECK(why.find("cannot remove") != std::string::npos);
    CHECK(why.find("the local session was not removed") != std::string::npos);
    ::chmod(victim.c_str(), 0755);
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, an_unchanged_generation_that_has_an_export_is_not_exported_again_by_the_policy) {
    Rig r("kvflash_s3r3_reexport", "{}");
    r.start("{\"quota_gib\": 1000}");
    const std::string out = fresh_root("kvflash_s3r3_reexport_out");
    std::string err;
    MUST(r.st->set_policy("{\"quota_gib\": 1000, \"action\": \"export\", \"after_export\": \"keep\", \"max_age_days\": 1, \"export_path\": \"" + out + "\"}", err));
    const std::string a = r.add("sid-a", r.now - 3 * 86400);
    const auto first = r.st->tick();
    REQUIRE(first.size() == 1);
    REQUIRE(r.st->export_session(r.st->claim_of(first[0].key), out).ok());
    CHECK(r.st->tick().empty());   // the same generation is already exported and verified
    CHECK(r.st->tick().empty());
    CHECK(names(out) == (std::set<std::string>{a + ".g1"}));
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, a_root_scan_that_fails_is_retried_and_never_trusted_as_a_short_count) {
    uint64_t n = 7;
    CHECK(!kvflash_tree_bytes_checked("/nonexistent-kvflash-s3r3", n));   // an unreadable root is unknown, not 0
    std::atomic<int> calls{0};
    Rig r("kvflash_s3r3_scan", "{}");
    r.env.scan_root = [&calls](const std::string &, uint64_t & out) { out = 123; return calls.fetch_add(1) >= 1; };   // fails once, then reads
    r.start("{\"quota_gib\": 1000}");
    uint64_t id = 0;
    std::string why;
    CHECK(r.st->reserve(1000, id, why));
    CHECK(calls.load() == 2);
    r.st->release_reservation(id);
    calls = -1000;   // never reads
    CHECK(!r.st->reserve(1000, id, why));
    CHECK(why.find("could not be read") != std::string::npos);
}

// ── round 4: the persistent export marker, unknown occupancy, the scan ──────────────────────

namespace {
// an aged session under an export policy; the policy's export of it, the way the steward does it
struct ExportRig {
    Rig r;
    std::string out;
    explicit ExportRig(const char * name, const char * after = "keep", const std::function<void(Rig &)> & pre = {}) : r(name, "{}"), out(fresh_root(std::string(name) + "_out")) {
        if (pre) pre(r);
        r.start("{\"quota_gib\": 1000}");
        set(after);
    }
    ~ExportRig() { fs::remove_all(out); }
    void set(const char * after) {
        std::string err;
        MUST(r.st->set_policy(std::string("{\"quota_gib\": 1000, \"action\": \"export\", \"after_export\": \"") + after + "\", \"max_age_days\": 1, \"export_path\": \"" + out + "\"}", err));
    }
    std::string aged(const std::string & sid = "sid-a") { return r.add(sid, r.now - 3 * 86400); }
    // one steward pass; every export it asks for is made (synchronously); returns how many
    int pass() {
        int n = 0;
        for (const auto & w : r.st->tick()) { if (r.st->export_session(r.st->claim_of(w.key), w.path).ok()) ++n; }
        return n;
    }
    size_t exports() const { return names(out).size(); }
};
}  // namespace

TEST_CASE(KvflashStoragePolicyFixture, a_recreated_session_at_the_same_key_and_generation_is_exported_by_the_policy_again) {
    ExportRig g("kvflash_s3r4_reuse");
    const std::string a = g.aged();
    CHECK(g.pass() == 1);
    CHECK(g.pass() == 0);   // the same head: not again
    REQUIRE(g.r.st->remove(g.r.st->claim_of(a)).ok());
    CHECK(g.aged() == a);   // the same key, the same generation, the same bytes
    CHECK(g.pass() == 1);   // a new session: the marker went with the old directory
    CHECK(g.exports() == 2);
}

TEST_CASE(KvflashStoragePolicyFixture, a_keep_export_then_a_policy_that_deletes_removes_the_local_copy_without_a_second_copy) {
    ExportRig g("kvflash_s3r4_keepdel");
    const std::string a = g.aged();
    REQUIRE(g.pass() == 1);
    g.set("delete");
    CHECK(g.pass() == 0);   // no copy: the removal alone
    CHECK(g.r.store.find_key(a) == nullptr);
    CHECK(g.exports() == 1);
}

TEST_CASE(KvflashStoragePolicyFixture, a_failed_local_removal_is_retried_alone_and_never_copies_again) {
    std::string dir;
    ExportRig g("kvflash_s3r4_rmretry", "delete", [&dir](Rig & r) {
        r.env.fsync_dir = [&dir](const std::string & d) { if (d == dir) ::chmod(dir.c_str(), 0555); return kvflash_fsync_dir(d); };   // right after the marker: the removal cannot unlink
    });
    const std::string a = g.aged();
    dir = g.r.store.dir(a);
    CHECK(g.pass() == 0);                             // the copy is made, the removal fails
    CHECK(g.exports() == 1);
    for (int i = 0; i < 3; ++i) CHECK(g.pass() == 0); // later passes retry the removal only
    CHECK(g.exports() == 1);
    CHECK(g.r.store.find_key(a) != nullptr);
    ::chmod(dir.c_str(), 0755);
    CHECK(g.pass() == 0);
    CHECK(g.r.store.find_key(a) == nullptr);          // the retry finally removes it
    CHECK(g.exports() == 1);
}

TEST_CASE(KvflashStoragePolicyFixture, a_restart_does_not_export_an_exported_head_again) {
    ExportRig g("kvflash_s3r4_restart");
    g.aged();
    REQUIRE(g.pass() == 1);
    g.set("keep");
    REQUIRE(g.r.st->compact(g.r.st->claim_of(g.r.store.entries().front().key)).status == 200);
    CHECK(!g.r.rescan().empty());   // the real session scan of the restart, which replays compaction unlinks
    g.r.st = std::make_unique<KvFlashStorage>(g.r.store, ID, RB, HEAD_MAX, g.r.env);   // a new process over the same root
    g.r.st->load();
    g.set("keep");
    CHECK(g.pass() == 0);
    CHECK(g.exports() == 1);
}

TEST_CASE(KvflashStoragePolicyFixture, a_new_head_after_an_export_is_exported_again) {
    ExportRig g("kvflash_s3r4_newhead");
    const std::string a = g.aged();
    REQUIRE(g.pass() == 1);
    write_text(g.r.store.dir(a) + "/head", slurp(g.r.store.dir(a) + "/head") + "x");   // a park commit rewrites the head
    CHECK(!g.r.st->tick().empty());
}

TEST_CASE(KvflashStoragePolicyFixture, an_unknown_store_size_refuses_the_park_with_its_own_reason_and_never_enters_the_cascade) {
    RoomRig g("kvflash_s3r4_unknown", "{\"reserve_gib\": 1, \"quota_gib\": 1000, \"action\":\"delete-oldest\"}", [&](Rig & r) {
        r.env.scan_root = [](const std::string &, uint64_t &) { return false; };
    });
    const std::string old = g.r.add("sid-old", 200, 20);
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));
    CHECK(why.find("could not be read") != std::string::npos);
    CHECK(why.find("make room") == std::string::npos);
    CHECK(g.r.store.find_key(old) != nullptr);   // nothing was deleted to make room for an unknown size
    g.r.now += 5 * 86400;
    g.r.st->tick();
    CHECK(g.r.store.find_key(old) != nullptr);   // and the steward leaves it alone too
}

TEST_CASE(KvflashStoragePolicyFixture, a_scan_that_cannot_stat_a_file_is_a_failed_scan) {
    Rig r("kvflash_s3r4_lstat", "{}");
    r.add("sid-a", 100, 4);
    uint64_t n = 0;
    CHECK(kvflash_tree_bytes_checked(r.root, n));
    CHECK(n > 0);
    uint64_t m = 7;
    CHECK(!kvflash_tree_bytes_checked(r.root, m, [](const char *, struct stat *) { errno = ENOENT; return -1; }));
    CHECK(m == 7);   // a failed scan leaves no size behind
    int calls = 0;
    CHECK(kvflash_tree_bytes_checked(r.root, m, [&calls](const char * p, struct stat * st) { return calls++ == 0 ? -1 : ::lstat(p, st); }));   // retried, then read whole
    CHECK(m == n);
    fs::create_symlink("/nonexistent-kvflash-dangling", r.root + "/dangling");
    CHECK(kvflash_tree_bytes_checked(r.root, m));   // a dangling link is not a regular file and not an error
}

TEST_CASE(KvflashStoragePolicyFixture, an_accepted_workers_507_stays_507_in_its_failure) {
    const std::string out = fresh_root("kvflash_s3r4_507_out");
    Rig r("kvflash_s3r4_507", "{}");
    const std::string a = r.add("sid-a", 100);
    KvFlashExportPlan plan;
    REQUIRE(r.st->export_begin(r.st->claim_of(a), out, plan).ok());
    r.st->export_commit(plan, KvFlashResult{507, "no space left"});   // the worker's copy ran out of room
    CHECK(r.st->action_failure_("export", a) == "507 no space left");
    fs::remove_all(out);
}

TEST_CASE(KvflashStoragePolicyFixture, restoring_a_session_from_disk_keeps_its_export_marker) {
    ExportRig g("kvflash_s3r4_restore");
    const std::string a = g.aged();
    REQUIRE(g.pass() == 1);
    const std::string dir = g.r.store.dir(a);
    REQUIRE(fs::exists(dir + "/exported"));
    KvFlashSlotHead h;
    REQUIRE(g.r.st->read_head(a, h));                       // what the slot restore loads
    kvflash_bundle_clean(dir, &h);                           // and then cleans: a park cut short, a torn tail
    CHECK(fs::exists(dir + "/exported"));                    // the marker is the steward's, not a park file
    CHECK(g.pass() == 0);
}

TEST_CASE(KvflashStoragePolicyFixture, a_compaction_keeps_the_export_marker_so_the_export_policy_never_copies_again) {
    ExportRig g("kvflash_s3r5_compact");
    const std::string a = g.aged();
    REQUIRE(g.pass() == 1);
    REQUIRE(g.r.st->compact(g.r.st->claim_of(a)).status == 200);
    CHECK(names(g.r.store.dir(a)) == (std::set<std::string>{"head", "meta.json", "compact.json", "exported"}));
    CHECK(g.pass() == 0);
    CHECK(g.exports() == 1);
}

TEST_CASE(KvflashStoragePolicyFixture, a_descriptor_that_lists_the_export_marker_leaves_it_in_place_at_startup_and_in_the_retry) {
    Rig r("kvflash_s3r5_descriptor");
    const std::string a = r.add("sid-a", 100);
    const std::vector<int32_t> hist = r.ent(a).history;
    r.unlink_fails = {kvflash_bundle_log_name(1)};
    REQUIRE(r.st->compact(r.st->claim_of(a)).status == 200);   // the log stays: the steward owes the unlink
    r.unlink_fails.clear();
    write_text(r.store.dir(a) + "/exported", "gen=1\nhead=x\npath=p\n");
    KvFlashCompactDescriptor d = descriptor_for(r.store.dir(a), hist, 1);
    d.unlink = {"exported", kvflash_bundle_log_name(1)};       // what an older build wrote
    write_text(r.store.dir(a) + "/compact.json", kvflash_compact_pack(d));
    r.st->tick();                                              // the retry
    CHECK(!fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(fs::exists(r.store.dir(a) + "/exported"));
    write_text(r.store.dir(a) + "/" + kvflash_bundle_log_name(1), "x");
    CHECK(!r.rescan().empty());                                // the startup replay
    CHECK(!fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
    CHECK(fs::exists(r.store.dir(a) + "/exported"));
}

TEST_CASE(KvflashStoragePolicyFixture, a_scan_that_fails_after_a_cascade_action_stops_the_cascade_with_the_unknown_size_reason) {
    int calls = 0;
    RoomRig g("kvflash_s3r5_cascade", "{\"reserve_gib\": 1, \"quota_gib\": 1000, \"action\":\"delete-oldest\"}");
    const std::string a = g.r.add("sid-a", 200, 20), b = g.r.add("sid-b", 300, 20);
    g.squeeze();
    g.r.env.scan_root = [&calls, &g](const std::string &, uint64_t & total) { if (calls++ > 0) return false; total = tree_bytes(g.r.root); return true; };
    g.r.start("{\"reserve_gib\": 1, \"quota_gib\": 1000, \"action\":\"delete-oldest\"}");
    uint64_t res = 0;
    std::string why;
    CHECK(!g.admit(res, why));
    CHECK(g.r.store.find_key(a) == nullptr);       // the first candidate went before the scan failed
    CHECK(g.r.store.find_key(b) != nullptr);       // the second one is not touched on an unknown size
    CHECK(why.find("could not be read") != std::string::npos);
}

TEST_CASE(KvflashStoragePolicyFixture, a_scan_that_fails_for_the_log_rewrite_reservation_does_not_compact) {
    int calls = 0;
    Rig r("kvflash_s3r5_tick", "{}");
    r.env.scan_root = [&calls, &r](const std::string &, uint64_t & total) { if (calls++ > 0) return false; total = tree_bytes(r.root); return true; };
    r.start("{\"reserve_gib\": 1, \"quota_gib\": 1000, \"log_dead_mib\": 1, \"action\":\"compact\"}");
    const std::string a = r.add("sid-a", r.now - 1000, 40, "disk", /*live*/ 10);
    r.ent(a).bytes_dead = 30 * RB;
    r.st->tick();                                   // the pass's own scan is good, the rewrite's reservation scan is not
    CHECK(r.ent(a).state == "disk");
    CHECK(fs::exists(r.store.dir(a) + "/" + kvflash_bundle_log_name(1)));
}

TEST_CASE(KvflashStoragePolicyFixture, an_unknown_size_skips_the_whole_pass_expiry_and_dead_log_reclamation_included) {
    bool fail = false;
    Rig r("kvflash_s3r5_unknown_pass", "{}");
    r.env.grace_s = 60;
    r.env.scan_root = [&fail, &r](const std::string &, uint64_t & total) { if (fail) return false; return kvflash_tree_bytes_checked(r.root, total); };
    r.start("{}");
    const auto keys = scenario(r, ",\"log_dead_mib\":1");
    r.st->tick();                                   // decisions are offered, the scan is good
    REQUIRE(r.st->pending().size() == 2);
    r.ent(keys[5]).bytes_dead = 30 * RB;            // dead-log debt
    fail = true;
    r.now += 61;                                    // the decisions have expired
    r.st->tick();
    CHECK(r.ent(keys[0]).state == "disk");          // an expired decision compacts only on a known size
    CHECK(r.ent(keys[1]).state == "disk");
    CHECK(r.ent(keys[5]).bytes_dead == 30 * RB);    // the log is not rewritten
    CHECK(fs::exists(r.store.dir(keys[5]) + "/" + kvflash_bundle_log_name(1)));
}

TEST_CASE(KvflashStoragePolicyFixture, an_export_path_with_a_control_character_is_refused_and_a_marker_must_start_with_its_own_fields) {
    ExportRig g("kvflash_s3r5_marker");
    const std::string a = g.aged();
    std::string err;
    CHECK(!g.r.st->set_policy("{\"action\": \"export\", \"export_path\": \"" + g.out + "\\ngen=1\\nhead=x\\n\"}", err));
    KvFlashExportPlan plan;
    CHECK(!g.r.st->export_begin(g.r.st->claim_of(a), g.out + "\ngen=1\nhead=x\n", plan).ok());
    REQUIRE(g.pass() == 1);
    const std::string m = slurp(g.r.store.dir(a) + "/exported");
    const size_t gen_at = m.find("gen=");
    write_text(g.r.store.dir(a) + "/exported", "path=spoof\n" + m);   // the real fields, but not first
    CHECK(gen_at == 0);
    CHECK(g.pass() == 1);                                               // no match: copied again
}
