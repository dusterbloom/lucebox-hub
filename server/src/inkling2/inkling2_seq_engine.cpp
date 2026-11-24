// Inkling2SeqEngine & Inkling2SlotKv implementation.

#include "inkling2_seq_engine.h"
#include "inkling2_graph.h"
#include "inkling2_kvflash.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <utility>

namespace luce::common {

// ─── Inkling2SlotKv (SlotKv v2) ──────────────────────────────────────────

uint32_t Inkling2SlotKv::compute_pool_blocks_(int max_ctx, size_t n_slots, uint32_t bs) {
    if (max_ctx <= 0 || n_slots == 0 || bs == 0) return 0;
    const uint64_t per_slot = ((uint64_t) max_ctx + bs - 1) / bs;
    const uint64_t total    = per_slot * (uint64_t) n_slots;
    return total <= std::numeric_limits<uint32_t>::max() ? (uint32_t) total : 0;
}

Inkling2SlotKv::Inkling2SlotKv(int max_ctx,
                               const std::vector<Inkling2Cache *> & caches,
                               uint32_t block_size)
    : pool_(compute_pool_blocks_(max_ctx, caches.size(), block_size),
            (uint32_t) caches.size(),
            block_size),
      caches_(caches) {}

SlotKvCaps Inkling2SlotKv::caps() const {
    SlotKvCaps c;
    c.parkable_offload        = true;
    c.sticky_sessions         = false;
    c.supports_prefix_store   = false;
    c.mutates_existing_blocks = false;
    c.dual_tier_tables        = false;
    for (const Inkling2Cache * cache : caches_) {
        if (cache && cache->kvflash && cache->kvflash->active()) {
            c.mutates_existing_blocks = true;
            if (cache->kvflash->tier15_active()) c.dual_tier_tables = true;
        }
    }
    return c;
}

PagedKvStatus Inkling2SlotKv::acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                               PagedKvSequenceHandle & h) {
    return pool_.acquire_reserved(id, cap, h);
}

PagedKvStatus Inkling2SlotKv::acquire_reserved(PagedKvRequestId id, uint32_t cap,
                                               const SlotAcquireOpts & opts,
                                               PagedKvSequenceHandle & h) {
    (void) opts;
    return pool_.acquire_reserved(id, cap, h);
}

PagedKvStatus Inkling2SlotKv::reserve_capacity(PagedKvSequenceHandle h, uint32_t cap) {
    return pool_.reserve_capacity(h, cap);
}

PagedKvAppendResult Inkling2SlotKv::append(PagedKvSequenceHandle h, uint32_t n, bool first_last) {
    return pool_.append(h, n, first_last);
}

PagedKvStatus Inkling2SlotKv::rollback_append(PagedKvSequenceHandle h, uint32_t n) {
    return pool_.rollback_append(h, n);
}

PagedKvStatus Inkling2SlotKv::clear(PagedKvSequenceHandle h) {
    return pool_.clear(h);
}

PagedKvStatus Inkling2SlotKv::release(PagedKvSequenceHandle h) {
    return pool_.release(h);
}

PagedKvStatus Inkling2SlotKv::owned_block_count(PagedKvSequenceHandle h, uint32_t & n) const {
    return pool_.owned_block_count(h, n);
}

bool Inkling2SlotKv::build_table_patch(PagedKvSequenceHandle h, uint32_t token_extent,
                                       uint32_t max_attn_blocks, SlotTablePatch & out,
                                       bool remote) const {
    out = {};
    PagedKvSequenceSnapshot snap;
    if (pool_.sequence(h, snap) != PagedKvStatus::Ok) return false;
    const uint32_t slot_id = h.slot;
    const uint32_t bs = std::max<uint32_t>(1, pool_.block_size());
    const uint32_t need = (token_extent + bs - 1) / bs;
    if (need > max_attn_blocks) return false;

    if (slot_id < caches_.size() && caches_[slot_id] &&
        caches_[slot_id]->kvflash && caches_[slot_id]->kvflash->active()) {
        const auto & kf = *caches_[slot_id]->kvflash;
        if (remote && !kf.tier15_active()) return false;
        const auto & bt = remote ? kf.bt_remote() : kf.bt_local();
        const uint32_t n_copy = std::min<uint32_t>(need, (uint32_t) bt.size());
        out.first_block = 0;
        out.blocks.assign(bt.begin(), bt.begin() + (ptrdiff_t) n_copy);
        return true;
    }

    if (remote) return false;
    if (need > snap.block_table.size()) return false;
    out.first_block = 0;
    out.blocks.resize(need);
    for (uint32_t i = 0; i < need; ++i) out.blocks[i] = (int32_t) snap.block_table[i];
    return true;
}

// ─── Inkling2SeqEngine ───────────────────────────────────────────────────

Inkling2SeqEngine::Inkling2SeqEngine(
        ggml_backend_t backend,
        const Inkling2Weights & weights,
        std::vector<Inkling2Cache *> caches,
        int max_ctx,
        int prefill_chunk,
        int verify_width,
        Inkling2SeqForwardFn custom_forward)
    : backend_(backend),
      weights_(weights),
      caches_(caches),
      slot_kv_(max_ctx, caches, (uint32_t) kInkling2KvFlashChunk),
      slots_(slot_kv_, max_ctx),
      state_adapter_(weights, caches, /*include_draft_ring=*/false),
      prefill_chunk_(std::max(1, prefill_chunk)),
      verify_width_(verify_width),
      custom_forward_(std::move(custom_forward)),
      rt_(caches.size()) {}

Inkling2SeqEngine::~Inkling2SeqEngine() {
    if (backend_) ggml_backend_synchronize(backend_);
}

bool Inkling2SeqEngine::token_is_eos(int32_t token) const {
    return token == weights_.eos_id || token == weights_.eos_chat_id;
}

int Inkling2SeqEngine::max_decode_tokens() const {
    if (verify_width_ == 1 || !weights_.dspark.loaded) return 1;
    const int cap = std::min(kInkling2MaxVerify, weights_.dspark.block_size);
    return (verify_width_ >= 2 && verify_width_ <= cap) ? verify_width_ : cap;
}

SeqEngine::AdmitResult Inkling2SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    return admit_bounded(request_id, prompt, sampler, 0, {});
}

SeqEngine::AdmitResult Inkling2SeqEngine::admit_bounded(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        int max_new_tokens,
        const AdmitMeta & meta) {
    (void) max_new_tokens; (void) meta;
    if (prompt.empty()) {
        AdmitResult r;
        r.status = AdmitResult::Status::failed;
        r.error  = "inkling2 prompt is empty";
        return r;
    }
    const int vocab = weights_.n_vocab_unpadded;
    for (int32_t tok : prompt) {
        if (tok < 0 || (vocab > 0 && tok >= vocab)) {
            AdmitResult r;
            r.status = AdmitResult::Status::failed;
            r.error  = "inkling2 prompt contains an invalid token id";
            return r;
        }
    }

    AdmitResult r = slots_.admit(request_id, prompt, sampler);
    if (r.status != AdmitResult::Status::admitted) return r;

    const int s = r.slot;
    if (s < 0 || s >= (int) caches_.size() || !caches_[(size_t) s]) {
        slots_.retire(s);
        r.status = AdmitResult::Status::failed;
        r.error  = "inkling2 slot cache unavailable";
        return r;
    }

    // Reserve full max_context on this slot's bookkeeping handle so concurrent decoders never starve.
    if (slot_kv_.reserve_capacity(slots_.slot(s).handle, (uint32_t) max_context()) != PagedKvStatus::Ok) {
        slots_.retire(s);
        r.status = AdmitResult::Status::busy;
        r.error  = "inkling2 slot reservation unavailable";
        return r;
    }

    Inkling2Cache & cache = *caches_[(size_t) s];
    if (!custom_forward_) {
        reset_inkling2_cache(cache);
    }

    SlotRuntime & sr = rt_[(size_t) s];
    sr = SlotRuntime{};
    sr.prompt = prompt;
    if (cache.kvflash && cache.kvflash->active()) {
        sr.rewind_pos = cache.kvflash->plan_mode4_rewind(prompt, -1);
        sr.q_cap_start = (sr.rewind_pos >= 0)
            ? std::max(sr.rewind_pos, (int) prompt.size() - cache.kvflash->select_queries())
            : (int) prompt.size();
        cache.kvflash->clear_queries();
    }
    return r;
}

StepPlanLimits Inkling2SeqEngine::step_plan_limits(int decode_rows) const {
    StepPlanLimits lim;
    lim.max_prefill_sequences           = std::min(slot_count(), (decode_rows > 0) ? 1 : 2);
    lim.max_prefill_tokens_per_sequence = prefill_chunk_;
    lim.max_prefill_tokens_total        = prefill_chunk_ * lim.max_prefill_sequences;
    lim.prefill_allocation_quantum      = std::min(64, prefill_chunk_);
    return lim;
}

bool Inkling2SeqEngine::reserve_decode(const StepPlan & plan) {
    std::vector<int> growth((size_t) slot_count(), 0);
    for (const StepInput & in : plan.decode) {
        if (in.slot < 0 || in.slot >= slot_count()) return false;
        growth[(size_t) in.slot] = 1;
    }
    return slots_.reserve_decode(growth);
}

std::string Inkling2SeqEngine::validate_plan_(const StepPlan & plan) const {
    const int n = slot_count();
    std::vector<uint8_t> seen_dec((size_t) n, 0);
    std::vector<uint8_t> seen_pre((size_t) n, 0);

    for (const StepInput & in : plan.decode) {
        if (in.slot < 0 || in.slot >= n) return "decode slot out of range";
        if (in.token < 0) return "decode input token is negative";
        if (seen_dec[(size_t) in.slot]) return "duplicate decode slot";
        if (!slots_.slot(in.slot).decoding()) return "decode slot is not in decode phase";
        seen_dec[(size_t) in.slot] = 1;
    }
    // Every live decoding slot must be present in plan.decode
    for (int s = 0; s < n; ++s) {
        if (slots_.slot(s).decoding() && !seen_dec[(size_t) s]) {
            return "plan omitted an active decoding slot";
        }
    }

    const StepPlanLimits lim = step_plan_limits((int) plan.decode.size());
    if ((int) plan.prefills.size() > lim.max_prefill_sequences) {
        return "prefill cohort exceeds max_prefill_sequences";
    }
    int total_pre = 0;
    for (const PrefillSlice & sl : plan.prefills) {
        if (sl.slot < 0 || sl.slot >= n) return "prefill slot out of range";
        if (sl.max_tokens <= 0 || sl.max_tokens > lim.max_prefill_tokens_per_sequence) {
            return "prefill slice exceeds per-sequence token limit";
        }
        if (seen_pre[(size_t) sl.slot] || seen_dec[(size_t) sl.slot]) {
            return "duplicate or decoding slot in prefill plan";
        }
        if (!slots_.slot(sl.slot).prefilling()) return "prefill slot is not in prefill phase";
        seen_pre[(size_t) sl.slot] = 1;
        total_pre += sl.max_tokens;
    }
    if (total_pre > lim.max_prefill_tokens_total) {
        return "prefill slices exceed max_prefill_tokens_total";
    }
    return {};
}

Inkling2SeqStepResult Inkling2SeqEngine::run_forward_(
        int slot, const int32_t * tokens, int n_tokens, int pos0,
        bool need_logits, int capture_q_rows) {
    if (custom_forward_) {
        return custom_forward_(slot, tokens, n_tokens, pos0, need_logits, capture_q_rows);
    }
    Inkling2SeqStepResult out;
    Inkling2Cache & cache = *caches_[(size_t) slot];
    Inkling2ForwardOut r = inkling2_forward(
        backend_, weights_, cache, tokens, n_tokens, pos0,
        /*verify=*/false, /*need_logits=*/need_logits,
        /*capture_taps_for_dspark=*/weights_.dspark.loaded,
        /*allow_split=*/true, /*fast_sample_temp=*/0.0f,
        /*capture_q_rows=*/capture_q_rows);
    if (!r.ok) {
        out.error = "inkling2_forward failed";
        return out;
    }
    out.ok = true;
    if (!r.greedy.empty()) out.greedy_token = r.greedy.back();
    if (need_logits && !r.logits.empty()) {
        const int nv = weights_.n_vocab_unpadded;
        out.logits.assign(r.logits.end() - nv, r.logits.end());
    }
    return out;
}

int32_t Inkling2SeqEngine::sample_slot_(SeqSlot & seq, const Inkling2SeqStepResult & r, bool is_greedy) {
    if (is_greedy && r.greedy_token >= 0) return r.greedy_token;
    if (!r.logits.empty()) {
        const auto pen_hist = seq.penalty_history();
        return (int32_t) sample_logits(r.logits.data(), (int) r.logits.size(), seq.sampler, pen_hist, seq.rng);
    }
    return r.greedy_token;
}

bool Inkling2SeqEngine::try_solo_dspark_step_(const StepInput & in, DecodeOutput & out) {
    if (custom_forward_ || !weights_.dspark.loaded || verify_width_ == 1 || !in.allow_speculation) {
        return false;
    }
    SeqSlot & seq = slots_.slot(in.slot);
    if (!sample_logits_is_argmax(seq.sampler)) return false; // greedy solo fast path

    Inkling2Cache & cache = *caches_[(size_t) in.slot];
    const int pos = seq.cur_pos;
    const int rem_ctx = max_context() - pos - 1;
    const int max_vw = std::min(kInkling2MaxVerify, weights_.dspark.block_size);
    const int vw = (verify_width_ >= 2 && verify_width_ <= max_vw) ? verify_width_ : max_vw;
    const int k_draft = std::min(vw - 1, rem_ctx);
    if (k_draft < 1) return false;

    std::vector<int32_t> drafts;
    if (!inkling2_dspark_draft(backend_, weights_, cache, in.token, pos, k_draft, drafts, 0) || drafts.empty()) {
        return false;
    }
    const int n_d = (int) drafts.size();
    std::vector<int32_t> verify_ids(1 + (size_t) n_d);
    verify_ids[0] = in.token;
    for (int i = 0; i < n_d; ++i) verify_ids[1 + (size_t) i] = drafts[(size_t) i];

    Inkling2ForwardOut vr = inkling2_forward(
        backend_, weights_, cache, verify_ids.data(), 1 + n_d, pos,
        /*verify=*/true, /*need_logits=*/false, /*capture_taps=*/true,
        /*allow_split=*/(weights_.expert_backend && weights_.split_exp > 0));
    if (!vr.ok || (int) vr.greedy.size() < 1 + n_d) return false;

    int n_acc = 0;
    int32_t final_tok = vr.greedy[0];
    std::vector<int32_t> committed;
    for (int i = 0; i < n_d; ++i) {
        const int32_t tgt = vr.greedy[(size_t) i];
        if (tgt == drafts[(size_t) i] && !token_is_eos(tgt)) {
            committed.push_back(tgt);
            ++n_acc;
            final_tok = vr.greedy[(size_t) (i + 1)];
        } else {
            final_tok = tgt;
            break;
        }
    }
    const int retained = n_acc + 1;
    if (!inkling2_verify_rollback(backend_, weights_, cache, pos, retained)) return false;
    if (ggml_tensor * taps = cache.verify_ws[1 + n_d].taps_out) {
        inkling2_dspark_commit_taps(backend_, weights_, cache, taps, pos, retained);
    }

    auto app = slots_.append_tokens(in.slot, verify_ids.data(), retained);
    if (!app.ok) return false;
    slots_.commit_step(in.slot);

    out.slot             = in.slot;
    out.committed_tokens = std::move(committed);
    out.token            = final_tok;
    out.spec_offered     = n_d;
    out.spec_accepted    = n_acc;
    out.failed           = false;
    return true;
}

SeqEngine::StepResult Inkling2SeqEngine::step(const StepPlan & plan) {
    StepResult res;
    if (plan.decode.empty() && plan.prefills.empty()) {
        return res;
    }

    const std::string val_err = validate_plan_(plan);
    if (!val_err.empty()) {
        res.error = val_err;
        return res;
    }

    // 1. Advance scheduled prefill slices (each slot runs on its own Inkling2Cache + KVFlash pool)
    for (const PrefillSlice & sl : plan.prefills) {
        const int s = sl.slot;
        SeqSlot & seq = slots_.slot(s);
        SlotRuntime & sr = rt_[(size_t) s];
        Inkling2Cache & cache = *caches_[(size_t) s];
        auto * kf = (cache.kvflash && cache.kvflash->active()) ? cache.kvflash.get() : nullptr;

        const int prompt_len = seq.prompt_len;
        const int pos0 = seq.cur_pos;
        int n = std::min(sl.max_tokens, prompt_len - pos0);
        if (sr.rewind_pos > pos0 && sr.rewind_pos < pos0 + n) {
            n = sr.rewind_pos - pos0;
        }
        const int end = pos0 + n;
        const bool is_last = (end == prompt_len);
        const bool is_greedy = sample_logits_is_argmax(seq.sampler);
        const int cap_q = (kf && sr.rewind_pos >= 0 && !sr.rewind_done && end > sr.q_cap_start)
            ? std::min(n, end - sr.q_cap_start) : 0;

        auto pc = slots_.append_prefill(s, n);
        if (!pc.ok) {
            res.error = "inkling2 append_prefill failed";
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        Inkling2SeqStepResult fr = run_forward_(
            s, sr.prompt.data() + pos0, n, pos0,
            /*need_logits=*/is_last && !is_greedy, cap_q);
        if (!fr.ok) {
            res.error = fr.error.empty() ? "inkling2 prefill forward failed" : fr.error;
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        // Save SWA+ShortConv rewind checkpoint at R if KVFlash Mode-4 is active
        if (kf && sr.rewind_pos > 0 && end == sr.rewind_pos && !sr.rewind_done && !custom_forward_) {
            kf->save_rewind_snap(cache, weights_, sr.rewind_pos);
        }

        // Mode-4 exact recall + turn rewind right at the end of prefill
        if (is_last && kf && sr.rewind_pos > 0 && !sr.rewind_done && kf->has_rewind_snap() && !custom_forward_) {
            sr.rewind_done = true;
            if (kf->mode4_score_and_recall(sr.rewind_pos, prompt_len)) {
                kf->restore_rewind_snap(cache, weights_);
                int rpos = sr.rewind_pos;
                while (rpos < prompt_len) {
                    const int rend = std::min(prompt_len, rpos + prefill_chunk_);
                    const int rn = rend - rpos;
                    const bool rlast = (rend == prompt_len);
                    fr = run_forward_(s, sr.prompt.data() + rpos, rn, rpos, rlast && !is_greedy, 0);
                    if (!fr.ok) {
                        kf->clear_rewind_snap();
                        res.error = "inkling2 Mode-4 rewind prefill failed";
                        res.decode.clear();
                        res.prefills.clear();
                        return res;
                    }
                    rpos = rend;
                }
            }
            kf->clear_rewind_snap();
        }

        PrefillOutput po;
        po.slot = s;
        if (is_last) {
            slots_.commit_prefill(s);
            po.status = PrefillOutput::Status::completed;
            po.token  = sample_slot_(seq, fr, is_greedy);
        } else {
            po.status = PrefillOutput::Status::advanced;
        }
        res.prefills.push_back(std::move(po));
    }

    // 2. Advance live decode cohort
    const bool solo_spec_eligible = plan.prefills.empty() && plan.decode.size() == 1;
    for (const StepInput & in : plan.decode) {
        DecodeOutput dout;
        dout.slot = in.slot;

        if (solo_spec_eligible && try_solo_dspark_step_(in, dout)) {
            res.decode.push_back(std::move(dout));
            continue;
        }

        SeqSlot & seq = slots_.slot(in.slot);
        const int pos = seq.cur_pos;
        const bool is_greedy = sample_logits_is_argmax(seq.sampler);

        auto app = slots_.append_token(in.slot, in.token);
        if (!app.ok) {
            res.error = "inkling2 decode append_token failed";
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        Inkling2SeqStepResult fr = run_forward_(
            in.slot, &in.token, 1, pos, /*need_logits=*/!is_greedy, /*capture_q_rows=*/0);
        if (!fr.ok) {
            slots_.rollback_step(in.slot);
            res.error = fr.error.empty() ? "inkling2 decode forward failed" : fr.error;
            res.decode.clear();
            res.prefills.clear();
            return res;
        }

        slots_.commit_step(in.slot);
        dout.token  = sample_slot_(seq, fr, is_greedy);
        dout.failed = false;
        res.decode.push_back(std::move(dout));
    }

    return res;
}

size_t Inkling2SeqEngine::kv_offload_capacity() const {
    return state_adapter_.slab_bytes() * (size_t) std::max(0, slot_count() - 1);
}

SeqEngine::KvOffloadState Inkling2SeqEngine::kv_offload_state(int slot) const {
    KvOffloadState st;
    if (slot < 0 || slot >= slot_count()) return st;
    const SeqSlot & seq = slots_.slot(slot);
    st.parked    = seq.parked();
    st.recompute = seq.recomputing();
    st.bytes     = rt_[(size_t) slot].parked_slab.size();
    return st;
}

bool Inkling2SeqEngine::offload_kv(int slot, size_t max_bytes, std::string & error) {
    if (slot < 0 || slot >= slot_count()) { error = "invalid slot"; return false; }
    const size_t need = state_adapter_.slab_bytes();
    if (max_bytes > 0 && need > max_bytes) { error = "slab exceeds budget"; return false; }
    SlotStateSlab slab;
    if (!state_adapter_.save_slot_state(slot, slab)) {
        error = "failed to save Inkling2 slot state slab";
        return false;
    }
    if (!slots_.detach_kv(slot)) {
        error = "SeqSlotManager::detach_kv failed";
        return false;
    }
    rt_[(size_t) slot].parked_slab = std::move(slab);
    return true;
}

bool Inkling2SeqEngine::restore_kv(int slot, std::string & error) {
    if (slot < 0 || slot >= slot_count()) { error = "invalid slot"; return false; }
    SeqSlot & seq = slots_.slot(slot);
    if (seq.recomputing()) {
        if (!slots_.resume_recompute(slot)) return false;
        if (!custom_forward_) state_adapter_.reset_slot_state(slot);
        rt_[(size_t) slot].prompt = seq.sample_history;
        rt_[(size_t) slot].rewind_done = false;
        return true;
    }
    if (!slots_.attach_kv(slot)) return false;
    if (!rt_[(size_t) slot].parked_slab.empty()) {
        if (!state_adapter_.restore_slot_state(slot, rt_[(size_t) slot].parked_slab)) {
            error = "failed to restore Inkling2 slot state slab";
            return false;
        }
        rt_[(size_t) slot].parked_slab.clear();
    }
    return true;
}

bool Inkling2SeqEngine::evict_kv(int slot, int32_t pending_token, std::string & error) {
    if (slot < 0 || slot >= slot_count()) { error = "invalid slot"; return false; }
    if (!slots_.evict_for_recompute(slot, pending_token)) {
        error = "SeqSlotManager::evict_for_recompute failed";
        return false;
    }
    rt_[(size_t) slot].parked_slab.clear();
    return true;
}

bool Inkling2SeqEngine::kv_restore_feasible(int slot) const {
    return slot >= 0 && slot < slot_count() && slots_.kv_restore_feasible(slot);
}

void Inkling2SeqEngine::retire(int slot) {
    if (slot < 0 || slot >= slot_count()) return;
    slots_.retire(slot);
    rt_[(size_t) slot] = SlotRuntime{};
    if (!custom_forward_ && caches_[(size_t) slot]) {
        reset_inkling2_cache(*caches_[(size_t) slot]);
    }
}

} // namespace luce::common
