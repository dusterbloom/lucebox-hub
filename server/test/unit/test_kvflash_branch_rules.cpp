// Branch-point rules for the two-rung ladder, settled on a pure model before any engine change (trackc/b4_replay/RULES.md).
// A tiny ladder simulator runs (prompt end, lcp) turns through the real pure functions (kvflash_resume_point, kvflash_rungs_after_resume /
// _after_rewind, kvflash_rungs_after_ckpt as on sessions-int 0eea61b7b) and prints the resume L of every turn for every candidate rule.
// Positions come from the 3090 replay (trackc/b4_replay/table.md): a turn's header rung sits at prompt-7, the parked history is prompt+2,
// an extension of the previous turn's prompt matches it up to prompt-4 ("ext"), a sibling of an earlier turn matches up to that turn's prompt end ("sib").
// No ggml, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_turn.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashBranchRulesFixture {};

enum class Rule { base, v1, v2, v3, v4, v5 };
const char * rule_name(Rule r) {
    switch (r) { case Rule::base: return "base"; case Rule::v1: return "v1"; case Rule::v2: return "v2"; case Rule::v3: return "v3"; case Rule::v4: return "v4"; default: return "v5"; }
}
constexpr Rule kRules[] = {Rule::base, Rule::v1, Rule::v2, Rule::v3, Rule::v4, Rule::v5};

struct Turn { std::string name; int p; int lcp; };   // lcp < 0: the session's first turn (cold)

// v1 = branch-rung 497d1babc, copied: the rung a rewind resumed at is protected as the trunk until a whole-history resume.
int v1_after_resume(int branch, int L, int history, int ckpt, int trunk) {
    const bool alive = branch > 0 && (branch == ckpt || branch == trunk);
    if (L >= history) return alive && L > branch ? -1 : branch;
    return alive && L >= branch ? branch : L;
}
KvFlashRungs v1_after_ckpt(int ckpt, int trunk, int p, int protect) {
    KvFlashRungs r;
    r.ckpt = p;
    if (protect > 0 && trunk == protect && trunk < p) r.trunk = trunk;
    else if (ckpt > 0 && ckpt < p) r.trunk = ckpt;
    else if (trunk > 0 && trunk < p) r.trunk = trunk;
    return r;
}

struct Sim {
    Rule rule;
    int H = 0, ck = -1, tr = -1;   // history, checkpoint rung, trunk rung (the two-rung ladder)
    int branch = -1;               // v1, v2: the protected rung, pins the trunk
    int extra = -1;                // v3: the separate branch-point rung; v4: the third ladder rung; v5: the third slot (pinned or the demoted trunk)
    bool pinned = false;           // v5: the third slot holds a branch-point rung (an event happened) and is not demoted-into

    // v5 is the real rule (kvflash_slot_turn.h): the engine calls the same two functions.
    int step_v5(const Turn & t) {
        const int q = t.p - 7;
        int L = -1;
        if (t.lcp >= 0 && H > 0) L = kvflash_resume_point(t.lcp, H, ck, tr, t.p, extra);
        if (L <= 0) { *this = Sim{rule}; H = t.p + 2; ck = q; return 0; }
        const KvFlashLadderMove at_resume = kvflash_ladder_resume({ck, tr, extra, pinned}, L, H);
        const KvFlashLadder to = kvflash_ladder_ckpt(at_resume.to, q).to;
        ck = to.ckpt; tr = to.trunk; extra = to.extra; pinned = to.pinned; H = t.p + 2;
        return L;
    }

    int step(const Turn & t) {
        if (rule == Rule::v5) return step_v5(t);
        const int q = t.p - 7;     // this turn's header rung
        int L = -1;
        if (t.lcp >= 0 && H > 0) {
            L = kvflash_resume_point(t.lcp, H, ck, tr, t.p);
            if ((rule == Rule::v3 || rule == Rule::v4) && extra > 0 && t.lcp >= extra && extra < t.p) L = std::max(L, extra);
        }
        if (L <= 0) {              // cold: the rungs go with the rows
            *this = Sim{rule}; H = t.p + 2; ck = q; return 0;
        }
        const int c_pre = ck, H_pre = H;
        std::vector<int> rungs;    // every rung at or below L stays; the highest is the fallback checkpoint
        for (int r : {ck, tr, extra}) if (r > 0 && r <= L) rungs.push_back(r);
        if (L < H) {               // the rewind: rungs above L are gone (qwen35_seq_engine.cpp rewind block)
            const KvFlashRungs rw = kvflash_rungs_after_rewind(ck, tr, L);
            ck = rw.ckpt; tr = rw.trunk;
            if (rule == Rule::v3 || rule == Rule::v4) {   // the restore of an extra rung that is the resume point
                if (ck != L && extra == L) { ck = L; tr = -1; for (int r : rungs) if (r < L) tr = std::max(tr, r); }
            }
            if (extra > L) { extra = -1; pinned = false; }
            H = L;
        }
        switch (rule) {
        case Rule::v1: branch = v1_after_resume(branch, L, H_pre, ck, tr); break;
        case Rule::v2: if (L < c_pre) branch = L; break;
        case Rule::v3: if (L < c_pre) extra = L; break;
        default: break;
        }
        const KvFlashRungs rr = kvflash_rungs_after_resume(ck, tr, L);
        ck = rr.ckpt; tr = rr.trunk;
        if (rule == Rule::v4) {    // a third ladder rung: the old trunk is demoted instead of dropped
            std::vector<int> all = {q};
            for (int r : rungs) if (r < q) all.push_back(r);
            std::sort(all.begin(), all.end(), std::greater<int>()); all.erase(std::unique(all.begin(), all.end()), all.end());
            ck = all[0]; tr = all.size() > 1 ? all[1] : -1; extra = all.size() > 2 ? all[2] : -1;
        } else {
            const KvFlashRungs ra = (rule == Rule::v1 || rule == Rule::v2) ? v1_after_ckpt(ck, tr, q, branch) : kvflash_rungs_after_ckpt(ck, tr, q);
            ck = ra.ckpt; tr = ra.trunk;
        }
        H = t.p + 2;
        return L;
    }
};

std::vector<int> run(Rule rule, const std::vector<Turn> & turns) {
    Sim s{rule}; std::vector<int> out;
    for (const Turn & t : turns) out.push_back(s.step(t));
    return out;
}

struct Scenario { std::string name; std::vector<Turn> turns; };

// Positions of S1 and S3 are the replay's; the chain, the three branches and the linear chat extend them the same way.
std::vector<Scenario> scenarios() {
    std::vector<Scenario> v;
    const std::vector<Turn> s1 = {{"T1", 2054, -1}, {"T2", 3678, 2050}, {"P1", 3849, 3674}, {"P2", 3844, 3678}, {"X", 3867, 3840}, {"T3", 5291, 3678}};
    v.push_back({"S1", s1});
    v.push_back({"S3", {{"T1", 2067, -1}, {"T2", 5294, 2063}, {"X", 8502, 5290}, {"Y", 5447, 5294}, {"T3", 6915, 5294}}});
    auto chain = s1;   // after S1: T4 extends T3, the tool-check P3 extends T4, T5 returns to T4's prompt end
    chain.push_back({"T4", 6900, 5287}); chain.push_back({"P3", 6925, 6896}); chain.push_back({"T5", 8500, 6900});
    v.push_back({"S1+chain", chain});
    v.push_back({"3 branches", {{"T1", 2054, -1}, {"T2", 3678, 2050}, {"P1", 3849, 3674}, {"P2", 3844, 3678}, {"P3", 3850, 3678}, {"T3", 5291, 3678}}});
    // one branch then tool-check passes, with and without a second branch (the event of v3 needs the second branch, v4 needs at most two passes)
    v.push_back({"P1 X T3", {{"T1", 2054, -1}, {"T2", 3678, 2050}, {"P1", 3849, 3674}, {"X", 3872, 3845}, {"T3", 5291, 3678}}});
    v.push_back({"P1 P2 X X2 T3", {{"T1", 2054, -1}, {"T2", 3678, 2050}, {"P1", 3849, 3674}, {"P2", 3844, 3678}, {"X", 3867, 3840}, {"X2", 3890, 3863}, {"T3", 5291, 3678}}});
    std::vector<Turn> lin = {{"T1", 2054, -1}};
    for (int i = 2, p = 2054; i <= 6; ++i) { const int prev = p; p += 1620; lin.push_back({"T" + std::to_string(i), p, prev - 4}); }
    v.push_back({"linear T1-T6", lin});
    return v;
}
}  // namespace

TEST_CASE(KvflashBranchRulesFixture, per_turn_resume_L_of_every_rule_and_the_winner_is_at_least_the_base_on_every_turn) {
    for (const Scenario & sc : scenarios()) {
        std::printf("\n%-14s |", sc.name.c_str());
        for (const Turn & t : sc.turns) std::printf(" %6s", t.name.c_str());
        std::printf("\n");
        std::vector<std::vector<int>> res;
        for (Rule r : kRules) {
            res.push_back(run(r, sc.turns));
            std::printf("  %-12s |", rule_name(r));
            for (int L : res.back()) std::printf(" %6d", L);
            std::printf("\n");
        }
        const std::vector<int> & base = res[0];
        const std::vector<int> & v3 = res[3], & v4 = res[4], & v5 = res[5];
        for (size_t i = 0; i < base.size(); ++i) {
            CHECK(v3[i] >= base[i]);
            CHECK(v4[i] >= base[i]);
            CHECK(v5[i] >= base[i]);
            CHECK(v5[i] >= v3[i]);   // v5 is at least v3 and v4 on every turn of every scenario
            CHECK(v5[i] >= v4[i]);
        }
        if (sc.name == "linear T1-T6") for (Rule r : kRules) CHECK(run(r, sc.turns) == base);   // no branch, no change: every rule equals the base
    }
}

TEST_CASE(KvflashBranchRulesFixture, v3_v4_and_v5_reach_T3_at_T2s_header_rung_in_S1) {
    const std::vector<Scenario> sc = scenarios();
    const int t2_rung = 3678 - 7;
    CHECK(run(Rule::base, sc[0].turns).back() == 0);       // the base goes cold
    CHECK(run(Rule::v3, sc[0].turns).back() == t2_rung);
    CHECK(run(Rule::v4, sc[0].turns).back() == t2_rung);
    CHECK(run(Rule::v5, sc[0].turns).back() == t2_rung);
}

TEST_CASE(KvflashBranchRulesFixture, v1_and_v2_fall_below_the_base_somewhere) {
    int v1_below = 0, v2_below = 0;
    for (const Scenario & sc : scenarios()) {
        const std::vector<int> b = run(Rule::base, sc.turns), a = run(Rule::v1, sc.turns), c = run(Rule::v2, sc.turns);
        for (size_t i = 0; i < b.size(); ++i) { v1_below += a[i] < b[i]; v2_below += c[i] < b[i]; }
    }
    CHECK(v1_below > 0);   // the replay: S1 P2 and S3 Y, T3
    CHECK(v2_below > 0);   // the chain T4 -> P3 -> T5
}
