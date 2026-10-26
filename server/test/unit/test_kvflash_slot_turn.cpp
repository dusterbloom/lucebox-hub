// Pure unit tests for the C1.4 per-slot turn bookkeeping (kvflash_slot_turn.h):
// mode-4 plan on/off, first-pass cuts, per-step snapshot/capture flags and
// sticky-session matching. No ggml, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_turn.h"

#include <vector>

using namespace luce::common;

namespace {
struct KvflashSlotTurnFixture {};

constexpr int32_t kImStart = 7, kUser = 9, kCt = 64, kQ = 32, kU = 512, kT = 64;

// A prompt of n tokens whose last user turn starts at `user_at` (-1: none).
std::vector<int32_t> prompt_with_user(int n, int user_at) {
    std::vector<int32_t> p((size_t)n, 1);
    if (user_at >= 0 && user_at + 1 < n) { p[(size_t)user_at] = kImStart; p[(size_t)user_at + 1] = kUser; }
    return p;
}
}  // namespace

TEST_CASE(KvflashSlotTurnFixture, cuts_force_r_then_u_cuts_and_a_last_cut_of_at_least_t_queries) {
    // r at 100 lands on a cut end; 100..2020 is 1920 tokens: 512,512,512,384 >= 64 untouched
    CHECK((kvflash_slot_turn_cuts(0, 100, 2020, kU, kT) == std::vector<int>{100, 612, 1124, 1636, 2020}));
    // 0..520 from r: the 8-token last cut would miss queries; its predecessor shrinks to end - 64
    CHECK((kvflash_slot_turn_cuts(0, 0, 520, kU, kT) == std::vector<int>{456, 520}));
    // a turn shorter than T is one cut and captures what there is
    CHECK((kvflash_slot_turn_cuts(1000, 1010, 1040, kU, kT) == std::vector<int>{1010, 1040}));
    // r == kv_offset: no cut before r
    CHECK((kvflash_slot_turn_cuts(1000, 1000, 1600, kU, kT) == std::vector<int>{1512, 1600}));
    // r before kv_offset clamps to it; r at end gives a first pass with no replay tokens
    CHECK((kvflash_slot_turn_cuts(1000, 10, 1600, kU, kT) == std::vector<int>{1512, 1600}));
    CHECK((kvflash_slot_turn_cuts(0, 600, 600, kU, kT) == std::vector<int>{512, 600}));
    CHECK(kvflash_slot_turn_cuts(600, 600, 600, kU, kT).empty());
}

TEST_CASE(KvflashSlotTurnFixture, plan_runs_mode4_only_with_the_profile_and_a_pool_that_does_not_fit) {
    const auto prompt = prompt_with_user(5000, 4800);          // 79 chunks > 32 blocks
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.first_pass());
    CHECK(t.r == 4800);
    CHECK(t.end == 5000);
    CHECK(t.reserve_chunks == (5000 - 4800 + kCt - 1) / kCt + 1);
    CHECK(t.snapshot_at_r());
    CHECK(!t.cuts.empty());
    CHECK(t.cuts.back() == 5000);
    CHECK(std::find(t.cuts.begin(), t.cuts.end(), 4800) != t.cuts.end());

    const KvFlashSlotTurn off = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false);
    CHECK(!off.recall());
    CHECK(off.end == 5000);
    CHECK(off.cuts.empty());

    // the user turn starts inside the first chunk: nothing sealed to score, LRU turn
    const KvFlashSlotTurn early = kvflash_slot_turn_plan(prompt_with_user(5000, 40), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(!early.recall());
    CHECK(early.r == -1);
    // 20 chunks + reserve fit 32 blocks: LRU turn
    const KvFlashSlotTurn fits = kvflash_slot_turn_plan(prompt_with_user(1280, 1100), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(!fits.recall());
    CHECK(fits.r == -1);
}

TEST_CASE(KvflashSlotTurnFixture, plan_on_a_resumed_session_rewinds_to_the_new_user_turn_only) {
    auto prompt = prompt_with_user(6000, 5900);
    prompt[(size_t)100] = kImStart; prompt[(size_t)101] = kUser;   // an older user turn inside the session
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 5880, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.first_pass());
    CHECK(t.kv_offset == 5880);
    CHECK(t.r == 5900);
    CHECK((t.cuts == std::vector<int>{5900, 6000}));
    // no user turn past the resume point: R is the resume point, nothing to snapshot
    const KvFlashSlotTurn none = kvflash_slot_turn_plan(prompt_with_user(6000, 100), 5880, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(none.first_pass());
    CHECK(none.r == 5880);
    CHECK(!none.snapshot_at_r());
    // nothing to prefill
    CHECK(!kvflash_slot_turn_plan(prompt, 6000, kImStart, kUser, kCt, kQ, kU, kT, true).recall());
}

TEST_CASE(KvflashSlotTurnFixture, next_cut_follows_the_plan_in_the_first_pass_and_u_cuts_in_the_replay) {
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    int pos = 0;
    std::vector<int> ends;
    while (int n = kvflash_slot_turn_next_cut(t, pos, kU)) { pos += n; ends.push_back(pos); }
    CHECK(ends == t.cuts);
    CHECK(kvflash_slot_turn_next_cut(t, 5000, kU) == 0);
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);          // 4800..5000 in one cut
    CHECK(kvflash_slot_turn_next_cut(t, 4000, kU) == kvflash_next_cut(4000, 5000, kU) - 4000);
    KvFlashSlotTurn lru;
    lru.end = 1300;
    CHECK(kvflash_slot_turn_next_cut(lru, 0, kU) == kvflash_next_cut(0, 1300, kU));
}

// G-D: a turn that ends in prefill (max_tokens 1: no decode step) parks the state after its last prompt
// token. With a tail the turn's final cut leaves the last `tail` tokens to a cut of their own, so the engine
// can stash the state before them and park that far short.
TEST_CASE(KvflashSlotTurnFixture, a_tail_cut_leaves_the_last_tokens_to_a_cut_of_their_own_except_in_the_first_pass) {
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU, 1) == 200);      // the first pass keeps its capture cut whole
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU, 1) == 199);      // the replay stops one short of the end
    CHECK(kvflash_slot_turn_next_cut(t, 4999, kU, 1) == 1);        // and the last token is its own cut
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU, 0) == 200);     // off: unchanged
    CHECK(kvflash_slot_turn_next_cut(t, 4000, kU, 1) == kvflash_slot_turn_next_cut(t, 4000, kU));   // a cut that does not end the turn is untouched
    KvFlashSlotTurn lru;                                              // an LRU turn: U cuts, the last one short by a token
    lru.end = 1300;
    int pos = 0, cuts = 0, last = 0;
    while (int n = kvflash_slot_turn_next_cut(lru, pos, kU, 1)) { pos += n; ++cuts; last = n; }
    CHECK(pos == 1300);
    CHECK(last == 1);
    CHECK(cuts == 4);                                                  // 512, 512, 275, 1
    lru.end = 1;                                                       // a one-token turn is one cut
    CHECK(kvflash_slot_turn_next_cut(lru, 0, kU, 1) == 1);
    lru.end = 1300; lru.ckpt = 1297;                                   // the header cut still stops at the checkpoint, then 2 and 1
    pos = 0;
    std::vector<int> ends;
    while (int n = kvflash_slot_turn_next_cut(lru, pos, kU, 1)) { pos += n; ends.push_back(pos); }
    CHECK((ends == std::vector<int>{512, 1024, 1297, 1299, 1300}));
    // a tail of 4 (R's replay window): the last four tokens are one cut, whatever U cuts came before
    KvFlashSlotTurn t4;
    t4.end = 1300;
    pos = 0;
    ends.clear();
    while (int n = kvflash_slot_turn_next_cut(t4, pos, kU, 4)) { pos += n; ends.push_back(pos); }
    CHECK((ends == std::vector<int>{512, 1024, 1296, 1300}));
    t4.end = 3;                                                        // a turn shorter than the tail is one cut
    CHECK(kvflash_slot_turn_next_cut(t4, 0, kU, 4) == 3);
    t4.end = 1300; t4.ckpt = 1297;                                     // the header cut wins: 1297 is within the tail, so 3 tokens follow whole
    pos = 0; ends.clear();
    while (int n = kvflash_slot_turn_next_cut(t4, pos, kU, 4)) { pos += n; ends.push_back(pos); }
    CHECK((ends == std::vector<int>{512, 1024, 1297, 1300}));
}

TEST_CASE(KvflashSlotTurnFixture, the_step_ending_at_r_snapshots_and_the_step_ending_the_first_pass_captures) {
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    int pos = 0;
    int snaps = 0, caps = 0;
    while (int n = kvflash_slot_turn_next_cut(t, pos, kU)) {
        const KvFlashTurnStep s = kvflash_slot_turn_step(t, pos, n);
        if (s.snapshot) { snaps++; CHECK(pos + n == 4800); }
        if (s.capture) { caps++; CHECK(pos + n == 5000); CHECK(n >= kT); }
        pos += n;
    }
    CHECK(snaps == 1);
    CHECK(caps == 1);
    KvFlashSlotTurn replay = t;
    replay.phase = KvFlashTurnPhase::Replay;
    CHECK(!kvflash_slot_turn_step(replay, 4800, 200).capture);
    CHECK(!kvflash_slot_turn_step(replay, 4288, 512).snapshot);
}

TEST_CASE(KvflashSlotTurnFixture, pick_session_takes_the_longest_common_prefix_of_at_least_min_lcp_ties_to_the_newest) {
    const std::vector<int32_t> a = prompt_with_user(500, 100);
    std::vector<int32_t> b = a; b.resize(400); b[399] = 42;
    std::vector<int32_t> c = a; c.resize(500);
    const std::vector<KvFlashParkedSession> parked = {{0, &a, 1}, {1, &b, 2}, {2, &c, 3}};
    std::vector<int32_t> prompt = a; prompt.push_back(5);
    KvFlashSessionPick p = kvflash_pick_session(parked, prompt, 64);
    CHECK(p.slot == 2);                                              // a and c tie at 500; c is newer
    CHECK(p.lcp == 500);
    prompt = b; prompt.push_back(5);
    p = kvflash_pick_session(parked, prompt, 64);
    CHECK(p.slot == 1);
    CHECK(p.lcp == 400);
    CHECK(!kvflash_pick_session(parked, std::vector<int32_t>(50, 1), 64).exact());   // 50 < min_lcp
    CHECK(kvflash_pick_session(parked, std::vector<int32_t>(50, 1), 16).slot == 2);   // ties to newest
    CHECK(!kvflash_pick_session({}, prompt, 64).exact());
    CHECK(kvflash_oldest_session(parked) == 0);
    CHECK(kvflash_oldest_session({}) == -1);
}

TEST_CASE(KvflashSlotTurnFixture, the_checkpoint_is_the_last_im_start_and_cuts_stop_there_except_in_the_first_pass) {
    auto prompt = prompt_with_user(5000, 4800);
    prompt[4990] = kImStart;                                           // the assistant header of the generation prompt
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true);
    CHECK(t.ckpt == 4990);
    CHECK(t.first_pass());
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);            // the first pass keeps its capture cut whole
    CHECK(!kvflash_slot_turn_step(t, 4800, 200).ckpt);
    t.phase = KvFlashTurnPhase::Replay;
    // a recall turn's sticky checkpoint is its R snapshot (a first-pass state, the one the cold arm
    // has at R): the replay does not stop at the header and takes no snapshot of its own
    CHECK(kvflash_sticky_ckpt_pos(t) == t.r);
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);
    CHECK(!kvflash_slot_turn_step(t, 4800, 200).ckpt);
    CHECK(!kvflash_slot_turn_step(t, 4800, 190).ckpt);
    const KvFlashSlotTurn lru = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false);
    CHECK(lru.ckpt == 4990);
    CHECK(kvflash_sticky_ckpt_pos(lru) == 4990);                        // an LRU turn: the header, from the only pass
    int pos = 0, n_ckpt = 0;
    while (int n = kvflash_slot_turn_next_cut(lru, pos, kU)) { n_ckpt += kvflash_slot_turn_step(lru, pos, n).ckpt; pos += n; CHECK(n <= kU); }
    CHECK(pos == 5000);
    CHECK(n_ckpt == 1);
    // resumed at the header itself: nothing before it to checkpoint in this turn
    CHECK(kvflash_slot_turn_plan(prompt, 4990, kImStart, kUser, kCt, kQ, kU, kT, true).ckpt == -1);
    CHECK(kvflash_sticky_ckpt_pos(kvflash_slot_turn_plan(prompt, 4990, kImStart, kUser, kCt, kQ, kU, kT, false)) == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(300, -1), 0, kImStart, kUser, kCt, kQ, kU, kT, true).ckpt == -1);
}

TEST_CASE(KvflashSlotTurnFixture, resume_point_is_the_history_end_else_the_prompt_end_checkpoint_else_cold) {
    CHECK(kvflash_resume_point(4535, 4535, 4470, -1, 4674) == 4535);     // the reply re-tokenized as generated
    CHECK(kvflash_resume_point(4500, 4535, 4470, -1, 4674) == 4470);     // it did not: back to the prompt end
    CHECK(kvflash_resume_point(4535, 4535, 4470, -1, 4535) == 4470);     // nothing new past the history: replay the reply
    CHECK(kvflash_resume_point(4535, 4535, -1, -1, 4535) == -1);         // no checkpoint: cold
    CHECK(kvflash_resume_point(4000, 4535, 4470, -1, 4674) == -1);       // edited before the checkpoint: cold
    CHECK(kvflash_resume_point(0, 4535, 4470, -1, 4674) == -1);
    CHECK(kvflash_resume_point(10, 10, 10, -1, 20) == 10);
    CHECK(kvflash_resume_point(10, 0, 0, -1, 20) == -1);
}

TEST_CASE(KvflashSlotTurnFixture, need_chunks_is_what_the_next_cut_adds_past_the_allocated_chunks) {
    CHECK(kvflash_slot_need_chunks(5022, 142, kCt, 79) == 2);        // chunk 78 is partial; 79 and 80 are new
    CHECK(kvflash_slot_need_chunks(5164, 16, kCt, 81) == 0);         // decode inside chunk 80
    CHECK(kvflash_slot_need_chunks(5180, 16, kCt, 81) == 1);         // the next 16 tokens cross into chunk 81
    CHECK(kvflash_slot_need_chunks(0, 512, kCt, 0) == 8);            // a full first cut
    CHECK(kvflash_slot_need_chunks(100, 0, kCt, 2) == 0);
    CHECK(kvflash_slot_need_chunks(100, 10, 0, 2) == 0);
}

// C1.5 F1: two rungs per session, trunk below ckpt. The resume takes the highest rung the prompt
// covers (Rule B); a new checkpoint moves the older one to the trunk (Rule A).
TEST_CASE(KvflashSlotTurnFixture, resume_point_takes_the_highest_rung_the_prompt_covers) {
    // history 4535, ckpt 4470 (the header), trunk 4000 (the previous header)
    CHECK(kvflash_resume_point(4535, 4535, 4470, 4000, 4674) == 4535);   // extends the history: len
    CHECK(kvflash_resume_point(4500, 4535, 4470, 4000, 4674) == 4470);   // diverges in the reply: the header
    CHECK(kvflash_resume_point(4200, 4535, 4470, 4000, 4674) == 4000);   // diverges between the rungs: the trunk
    CHECK(kvflash_resume_point(3999, 4535, 4470, 4000, 4674) == -1);     // below both: cold
    CHECK(kvflash_resume_point(4535, 4535, 4470, 4000, 4535) == 4470);   // identical prompt: the rung below prompt_len
    CHECK(kvflash_resume_point(4470, 4535, 4470, 4000, 4470) == 4000);   // a prompt ending at the header: the trunk
    CHECK(kvflash_resume_point(4200, 4535, -1, 4000, 4674) == 4000);     // no ckpt, a trunk
    CHECK(kvflash_resume_point(4535, 4535, -1, -1, 4535) == -1);
    CHECK(kvflash_resume_point(4500, 4535, 4000, 4470, 4674) == 4470);   // the largest wins whatever the argument order
}

// G-D: park one token short. The slot keeps the state before its last fed token (the stash) and
// parks there, so the history is len-1 tokens and an identical prompt still has a token to prefill.
TEST_CASE(KvflashSlotTurnFixture, park_point_is_one_token_short_when_the_stash_holds_that_state) {
    CHECK(kvflash_park_point(4535, 4534, 4470, 4000) == 4534);   // the stash is the state before the last fed token
    CHECK(kvflash_park_point(4535, -1, 4470, 4000) == 4535);     // no stash (chain round, no decode step): park at len
    CHECK(kvflash_park_point(4535, 4530, 4470, 4000) == 4535);   // a stale stash (not len-1): park at len
    CHECK(kvflash_park_point(4535, 4534, 4535, 4000) == 4535);   // a rung at len: parking below it would orphan the rung
    CHECK(kvflash_park_point(4535, 4534, 4534, -1) == 4534);     // a rung at len-1 stays valid: it is the state at L
    CHECK(kvflash_park_point(1, 0, -1, -1) == 1);                // nothing below len-1 worth parking
    CHECK(kvflash_park_point(4535, 4531, 4470, 4000, 4) == 4531);   // a tail of 4: a state up to 4 tokens back is parked as it is
    CHECK(kvflash_park_point(4535, 4534, 4470, 4000, 4) == 4534);   // and a decode turn's one-back state too
    CHECK(kvflash_park_point(4535, 4530, 4470, 4000, 4) == 4535);   // more than the tail back: not a park point
    CHECK(kvflash_park_point(4535, 4531, 4470, 4000, 1) == 4535);   // tail 1 (the default): only len-1
    CHECK(kvflash_park_point(4535, 4531, 4532, 4000, 4) == 4535);   // a rung above the state: park at len
    CHECK(kvflash_park_point(4535, 4535, 4470, 4000, 4) == 4535);   // a stash at len is no stash
}

TEST_CASE(KvflashSlotTurnFixture, a_same_prompt_resumes_at_the_park_point_and_prefills_one_token) {
    const int len = 4535, P = kvflash_park_point(len, len - 1, 4470, 4000);
    // the prompt equals the history of len tokens; the session holds the first P of them
    const int L = kvflash_resume_point(P, P, 4470, 4000, len);
    CHECK(L == len - 1);
    CHECK(len - L == 1);                                          // exactly the pending token
    CHECK(kvflash_resume_point(P, P, -1, -1, len) == len - 1);    // with no rung at all (it was cold before)
    // without the stash the same prompt rewound to a rung (the G-D failure)
    CHECK(kvflash_resume_point(len, len, 4470, 4000, len) == 4470);
}

TEST_CASE(KvflashSlotTurnFixture, a_continuation_resumes_at_the_park_point_and_the_rung_order_holds_below_it) {
    const int len = 4535, P = kvflash_park_point(len, len - 1, 4470, 4000);
    CHECK(kvflash_resume_point(P, P, 4470, 4000, 4674) == P);     // re-rendered reply shares the history: resume at P, 140 new tokens
    CHECK(kvflash_resume_point(4500, P, 4470, 4000, 4674) == 4470);   // diverges in the reply: the header, as before
    CHECK(kvflash_resume_point(4200, P, 4470, 4000, 4674) == 4000);   // between the rungs: the trunk, as before
    CHECK(kvflash_resume_point(3999, P, 4470, 4000, 4674) == -1);     // below both: cold, as before
    const KvFlashRungs r = kvflash_rungs_after_resume(4470, 4000, P);  // at P both rungs stay
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
}

TEST_CASE(KvflashSlotTurnFixture, the_rungs_after_a_resume_are_the_one_at_L_and_the_one_below_it) {
    KvFlashRungs r = kvflash_rungs_after_resume(4470, 4000, 4535);       // at len: both kept
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
    r = kvflash_rungs_after_resume(4470, 4000, 4470);                    // at the ckpt: it is the fallback, the trunk stays
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
    r = kvflash_rungs_after_resume(4470, 4000, 4000);                    // at the trunk: it becomes the fallback, the ckpt is gone
    CHECK(r.ckpt == 4000); CHECK(r.trunk == -1);
    r = kvflash_rungs_after_resume(4470, -1, 4470);
    CHECK(r.ckpt == 4470); CHECK(r.trunk == -1);
    r = kvflash_rungs_after_resume(-1, -1, 4535);
    CHECK(r.ckpt == -1); CHECK(r.trunk == -1);
    r = kvflash_rungs_after_resume(4470, 4000, 4200);                    // L between the rungs (not a rung: the engine never does this): the trunk is the fallback
    CHECK(r.ckpt == 4000); CHECK(r.trunk == -1);
}

TEST_CASE(KvflashSlotTurnFixture, a_new_checkpoint_moves_the_older_one_to_the_trunk) {
    KvFlashRungs r = kvflash_rungs_after_ckpt(4470, 4000, 4990);         // q < p: q becomes the trunk, the old trunk goes
    CHECK(r.ckpt == 4990); CHECK(r.trunk == 4470);
    r = kvflash_rungs_after_ckpt(4470, 4000, 4470);                      // q == p: replaced, the trunk stays
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
    r = kvflash_rungs_after_ckpt(-1, 4000, 4470);                        // no ckpt: the trunk stays
    CHECK(r.ckpt == 4470); CHECK(r.trunk == 4000);
    r = kvflash_rungs_after_ckpt(-1, -1, 4470);
    CHECK(r.ckpt == 4470); CHECK(r.trunk == -1);
    r = kvflash_rungs_after_ckpt(4470, 4000, 4200);                      // q > p (after a truncation): replaced, a trunk below p stays
    CHECK(r.ckpt == 4200); CHECK(r.trunk == 4000);
    r = kvflash_rungs_after_ckpt(4470, 4000, 3000);
    CHECK(r.ckpt == 3000); CHECK(r.trunk == -1);
}

// C1.5 F7: on a recall turn the replay stops at the header and checkpoints there when end - R
// reaches LUCE_KVFLASH_END_RUNG_MIN; the first pass and its capture cut are unchanged.
TEST_CASE(KvflashSlotTurnFixture, the_end_rung_stops_the_replay_at_the_header_when_the_policy_allows_it) {
    auto prompt = prompt_with_user(5000, 4800);
    prompt[4990] = kImStart;                                           // the header inside the last U cut
    KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, 100);
    CHECK(t.first_pass());
    CHECK(t.end_rung);                                                 // end - r = 200 >= 100
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);            // the first pass keeps its capture cut whole
    CHECK(!kvflash_slot_turn_step(t, 4800, 200).ckpt);
    CHECK(kvflash_slot_turn_step(t, 4800, 200).capture);
    t.phase = KvFlashTurnPhase::Replay;
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 190);            // the replay stops at the header
    CHECK(kvflash_slot_turn_step(t, 4800, 190).ckpt);
    CHECK(kvflash_slot_turn_next_cut(t, 4990, kU) == 10);
    CHECK(!kvflash_slot_turn_step(t, 4990, 10).ckpt);
    CHECK(kvflash_slot_turn_next_cut(t, 4000, kU) == std::min(kvflash_next_cut(4000, 5000, kU), 4990) - 4000);
    CHECK(!kvflash_slot_turn_step(t, 4000, 512).ckpt);
    // the same turn with a higher threshold, or with the policy unset: today's replay
    for (int min : {201, 2048, -1}) {
        KvFlashSlotTurn off = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, min);
        off.phase = KvFlashTurnPhase::Replay;
        CHECK(!off.end_rung);
        CHECK(kvflash_slot_turn_next_cut(off, 4800, kU) == 200);
        CHECK(!kvflash_slot_turn_step(off, 4800, 190).ckpt);
    }
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, 0).end_rung);      // 0: always
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, 200).end_rung);    // at the threshold
    // no header after R (the question ends the prompt): nothing to rung
    CHECK(!kvflash_slot_turn_plan(prompt_with_user(5000, 4800), 0, kImStart, kUser, kCt, kQ, kU, kT, true, 0).end_rung);
    // an LRU turn ignores the policy: its header checkpoint is there anyway
    const KvFlashSlotTurn lru = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false, 0);
    CHECK(!lru.end_rung);
    CHECK(kvflash_slot_turn_step(lru, 4800, 190).ckpt);
}

TEST_CASE(KvflashSlotTurnFixture, mode5_plan_applies_the_drafters_picks_at_r_with_no_first_pass_and_no_r_snapshot) {
    const auto prompt = prompt_with_user(5000, 4800);          // 79 chunks > 32 blocks
    const KvFlashSlotTurn t = kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true);
    CHECK(!t.recall());                                         // one prefill: the LRU turn's checkpoint rule holds
    CHECK(t.apply_at == 4800);
    CHECK(t.reserve_chunks == (5000 - 4800 + kCt - 1) / kCt + 1);
    CHECK(t.cuts.empty());
    CHECK(!t.snapshot_at_r());
    CHECK(t.ckpt_pass());
    CHECK(!kvflash_slot_turn_step(t, 4500, 300).capture);
    CHECK(!kvflash_slot_turn_step(t, 4500, 300).snapshot);
    // the prefill stops at R, then runs plain cuts
    CHECK(kvflash_slot_turn_next_cut(t, 4700, kU) == 100);
    CHECK(kvflash_slot_turn_next_cut(t, 4800, kU) == 200);
    CHECK(!kvflash_slot_turn_apply_due(t, 4799));
    CHECK(kvflash_slot_turn_apply_due(t, 4800));
    CHECK(kvflash_slot_turn_apply_due(t, 4900));

    // resumed at R: due at once; no apply when the profile is off, the pool fits or nothing is sealed before R
    CHECK(kvflash_slot_turn_apply_due(kvflash_slot_turn_plan(prompt, 4800, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true), 4800));
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, false, -1, true).apply_at == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(1000, 900), 0, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true).apply_at == -1);
    CHECK(kvflash_slot_turn_plan(prompt_with_user(5000, 40), 0, kImStart, kUser, kCt, kQ, kU, kT, true, -1, true).apply_at == -1);
    // mode 4 plans never carry an apply point
    CHECK(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true).apply_at == -1);
    CHECK(!kvflash_slot_turn_apply_due(kvflash_slot_turn_plan(prompt, 0, kImStart, kUser, kCt, kQ, kU, kT, true), 5000));
}

TEST_CASE(KvflashSlotTurnFixture, park_cfg_tail_on_by_default_decode_stash_off_until_asked) {
    auto c = kvflash_park_cfg(true, nullptr, nullptr, nullptr);
    CHECK(c.tail_on); CHECK(!c.decode_on); CHECK(c.tail == 4); CHECK(c.any());
    c = kvflash_park_cfg(true, nullptr, "1", "1");
    CHECK(c.decode_on); CHECK(c.tail == 1);
    CHECK(!kvflash_park_cfg(true, nullptr, nullptr, "0").decode_on);
    CHECK(kvflash_park_cfg(true, nullptr, "999", nullptr).tail == 64);
    CHECK(kvflash_park_cfg(true, nullptr, "0", nullptr).tail == 1);
    c = kvflash_park_cfg(true, "0", nullptr, "1");           // PARK_SHORT=0 beats everything
    CHECK(!c.any());
    CHECK(!kvflash_park_cfg(false, nullptr, nullptr, "1").any());   // no sticky sessions, no parking
}
