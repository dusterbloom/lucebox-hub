// mtp_tree_runner.cpp — see mtp_tree_runner.h for contract.

#include "mtp_tree_runner.h"

#include "ddtree.h"
#include "dflash_target.h"
#include "mtp_chain_runner.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace dflash27b::mtp {

int env_tree_b(int hard_max) {
    const char * s = std::getenv("DFLASH27B_MTP_TREE_B");
    if (!s || !*s) return 1;
    int v = std::atoi(s);
    if (v <= 0) return 1;
    if (v > hard_max) v = hard_max;
    return v;
}

MtpTreeRunner::MtpTreeRunner(IMtpModule & mtp,
                             DFlashTarget & target,
                             const SamplerCfg & sampler,
                             int B,
                             int K)
    : mtp_(mtp), target_(target), sampler_cfg_(sampler),
      B_(std::max(1, B)), K_(std::max(1, K)) {
}

GenerateResult MtpTreeRunner::run(const GenerateRequest & req,
                                  const DaemonIO & io,
                                  int32_t last_prefill_token,
                                  int committed_pos,
                                  int gamma) {
    // B<=1 short-circuits to the chain runner.  This is the byte-identical
    // gate: the orchestrator dispatches into the tree runner unconditionally
    // when env-gated, and B=1 must produce the same tokens / KV trajectory
    // as the pre-tree path.
    if (B_ <= 1) {
        MtpChainRunner chain(mtp_, target_, sampler_cfg_);
        GenerateResult res = chain.run(req, io, last_prefill_token,
                                       committed_pos, gamma);
        const auto & cs = chain.stats();
        stats_.total_iters    = cs.total_iters;
        stats_.total_proposed = cs.total_proposed;
        stats_.total_accepted = cs.total_accepted;
        stats_.total_emitted  = cs.total_emitted;
        stats_.eos_hits       = cs.eos_hits;
        return res;
    }

    // B>=2 tree path requires NativeHeads (the arena slot routing surface
    // lives on INativeMtp::step_batch_arena).  Bail loud rather than silently
    // mis-route ExternalDrafter through chain — the caller asked for B>=2,
    // saying yes-but-quietly-no would mask the regression.
    if (mtp_.flavor() != MtpFlavor::NativeHeads) {
        GenerateResult res;
        res.error = "mtp_tree_runner: B>=2 requires NativeHeads flavor";
        return res;
    }
    auto & native = static_cast<INativeMtp &>(mtp_);

    GenerateResult result;
    const int n_gen = req.n_gen;
    if (n_gen <= 0) { result.ok = true; return result; }

    const int gamma_max = std::max(1, mtp_.max_gamma());
    if (gamma > gamma_max) gamma = gamma_max;
    if (gamma < 1)         gamma = 1;

    auto t0 = std::chrono::steady_clock::now();
    result.tokens.reserve(n_gen);

    int32_t cur_tok  = last_prefill_token;
    int     base_pos = committed_pos;

    // Per-iter arena reset is the module's responsibility — it owns the
    // arena tensor.  The runner only enforces the slot allocation contract
    // (path_id * gamma_max + depth) by passing (path_id, depth) honestly.
    struct ChainCaptureGuard {
        DFlashTarget & t;
        ~ChainCaptureGuard() { t.enable_chain_capture(false); }
    };
    target_.enable_chain_capture(true);
    ChainCaptureGuard guard{target_};

    bool hit_eos = false;
    bool tree_supported = true;  // flips false on first verify_tree() == false

    while ((int)result.tokens.size() < n_gen && !hit_eos) {
        const int remaining = n_gen - (int)result.tokens.size();
        const int g_iter    = std::min(gamma, remaining);

        // Per-iter arena reset — sibling slots from the previous iter must
        // not leak into this iter's FA read window.  No-op on B=1 (chain
        // short-circuit above) and on modules without an arena.
        native.reset_arena();

        // ── Propose B sibling chains of depth g_iter via arena-routed step ──
        // Layout: per_path_outs[path][depth] holds the StepOutput at that
        // (path, depth).  Each path's step_batch_arena writes to its own
        // arena slot row so paths don't poison each other's head_kv.
        std::vector<std::vector<StepOutput>> per_path_outs(B_);
        for (int p = 0; p < B_; ++p) per_path_outs[p].reserve(g_iter);

        // Depth 0: every path starts from cur_tok with its own arena row.
        // We pick the path-p token from the topk of path 0's depth-0 emission
        // (which IS the head's distribution conditioned on cur_tok); that
        // gives B distinct depth-0 sibling tokens with rank 0..B-1.
        std::vector<StepOutput> seed;
        if (!native.step_batch_arena(cur_tok, /*path_id=*/0, base_pos, /*depth=*/0,
                                     /*K=*/std::max(K_, B_), seed) ||
            seed.empty()) {
            result.ok = false;
            result.error = "tree_runner: seed step failed";
            return result;
        }
        // Pull B sibling tokens from depth-0 topk (rank 0..B-1).
        std::vector<int32_t> depth0_tokens(B_);
        if ((int)seed[0].topk_ids.size() >= B_) {
            for (int p = 0; p < B_; ++p) depth0_tokens[p] = seed[0].topk_ids[p];
        } else {
            // Fall back: identical sibling tokens — degenerate tree, but
            // semantically safe.  The accept loop will fold them.
            for (int p = 0; p < B_; ++p) depth0_tokens[p] = seed[0].draft_token;
        }
        // Stash path-0 depth-0 in slot 0; other paths get a fresh forward.
        per_path_outs[0].push_back(seed[0]);
        for (int p = 1; p < B_; ++p) {
            std::vector<StepOutput> tmp;
            if (!native.step_batch_arena(cur_tok, p, base_pos, 0,
                                          std::max(K_, B_), tmp) ||
                tmp.empty()) {
                result.ok = false;
                result.error = "tree_runner: sibling seed failed";
                return result;
            }
            // Force this path's sibling token to depth0_tokens[p] for the
            // child step input.  The path's own argmax may differ; we treat
            // the tree's path-p tokens as fixed by depth0 topk.
            per_path_outs[p].push_back(tmp[0]);
            per_path_outs[p].back().draft_token = depth0_tokens[p];
        }

        // Depths 1..g_iter-1: each path autoregresses from its previous draft.
        for (int d = 1; d < g_iter; ++d) {
            for (int p = 0; p < B_; ++p) {
                const int32_t parent = per_path_outs[p].back().draft_token;
                std::vector<StepOutput> tmp;
                if (!native.step_batch_arena(parent, p, base_pos, d, K_, tmp) ||
                    tmp.empty()) {
                    result.ok = false;
                    result.error = "tree_runner: path step failed";
                    return result;
                }
                per_path_outs[p].push_back(tmp[0]);
            }
        }

        // ── Build per-sibling topk arrays [L * B * K] for ddtree ──────────
        const int L = g_iter;
        std::vector<float>   tlp((size_t)L * B_ * K_, 0.0f);
        std::vector<int32_t> tid((size_t)L * B_ * K_, 0);
        for (int d = 0; d < L; ++d) {
            for (int p = 0; p < B_; ++p) {
                const auto & so = per_path_outs[p][d];
                const int have = std::min((int)so.topk_ids.size(), K_);
                for (int k = 0; k < have; ++k) {
                    const size_t i = (size_t)d * B_ * K_ + (size_t)p * K_ + k;
                    tlp[i] = so.topk_logprobs[k];
                    tid[i] = so.topk_ids[k];
                }
                // Fallback rank 0 for any missing topk (degenerate K=1 modules).
                if (have == 0) {
                    const size_t i = (size_t)d * B_ * K_ + (size_t)p * K_ + 0;
                    tid[i] = so.draft_token;
                    tlp[i] = so.draft_logit;
                }
            }
        }

        const int budget = std::max(1, L * B_);  // balanced B-ary cap
        DDTree tree = build_ddtree_tree(tlp.data(), tid.data(),
                                        L, B_, K_, budget);
        const int n_nodes = tree.n_nodes;

        // ── Verify via target.verify_tree ───────────────────────────────
        std::vector<int32_t> flat;
        flat.reserve((size_t)n_nodes + 1);
        flat.push_back(cur_tok);
        for (int i = 0; i < n_nodes; ++i) flat.push_back(tree.token_ids[i]);

        if (!target_.snapshot_kv()) {
            result.ok = false;
            result.error = "snapshot_kv";
            return result;
        }
        std::vector<int32_t> all_argmax;
        const bool ok_tree = tree_supported
            && target_.verify_tree(flat, tree, base_pos, all_argmax, nullptr);
        if (!ok_tree) {
            // Tree-verify unsupported on this target (or rejected this iter).
            // Restore + delegate the rest of decoding to the chain runner so
            // the request still completes.  This is a one-way switch for the
            // remainder of this run.
            target_.restore_kv();
            tree_supported = false;
            GenerateRequest sub = req;
            sub.n_gen = n_gen - (int)result.tokens.size();
            MtpChainRunner chain(mtp_, target_, sampler_cfg_);
            GenerateResult sub_res = chain.run(sub, io, cur_tok, base_pos, gamma);
            for (int32_t t : sub_res.tokens) result.tokens.push_back(t);
            const auto & cs = chain.stats();
            stats_.total_iters    += cs.total_iters;
            stats_.total_proposed += cs.total_proposed;
            stats_.total_accepted += cs.total_accepted;
            stats_.total_emitted  += cs.total_emitted;
            stats_.eos_hits       += cs.eos_hits;
            if (!sub_res.ok) {
                result.ok = sub_res.ok;
                result.error = sub_res.error;
            } else {
                result.ok = true;
            }
            auto t1 = std::chrono::steady_clock::now();
            result.decode_s = std::chrono::duration<double>(t1 - t0).count();
            return result;
        }

        stats_.total_proposed += n_nodes;

        // ── Walk the verified tree following target's argmax ────────────
        int next_tok = -1;
        int last_node = 0;
        std::vector<int> accepted = follow_verified_tree(
            tree, all_argmax.data(), next_tok, &last_node);
        const int accept_n = (int)accepted.size() - 1;  // exclude root

        // Rollback KV to the accepted path tail.  restore_kv_at_dfs handles
        // SSM + conv + KV; on failure fall back to full snapshot restore.
        if (!target_.restore_kv_at_dfs(accepted)) {
            if (!target_.restore_kv()) {
                result.ok = false;
                result.error = "restore_kv";
                return result;
            }
        }

        // ── Emit accepted prefix + bonus, capped at n_gen ───────────────
        const int total_this_iter = accept_n + 1;
        const int emit_cap = std::min(total_this_iter,
                                      n_gen - (int)result.tokens.size());
        int emitted = 0;
        for (int i = 0; i < accept_n && emitted < emit_cap; ++i) {
            const int32_t t = tree.token_ids[accepted[i + 1] - 1];
            result.tokens.push_back(t);
            if (req.stream) io.emit(t);
            emitted++;
            if (target_.is_eos(t)) { hit_eos = true; break; }
        }
        if (!hit_eos && emitted < emit_cap) {
            result.tokens.push_back(next_tok);
            if (req.stream) io.emit(next_tok);
            emitted++;
            if (target_.is_eos(next_tok)) hit_eos = true;
            cur_tok = next_tok;
        } else {
            cur_tok = result.tokens.empty() ? cur_tok : result.tokens.back();
        }

        base_pos += total_this_iter;
        stats_.total_iters    += 1;
        stats_.total_accepted += accept_n;
        stats_.total_emitted  += emitted;
    }

    if (hit_eos) stats_.eos_hits++;
    if (req.stream) io.emit(-1);

    auto t1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t1 - t0).count();
    result.ok = true;
    return result;
}

}  // namespace dflash27b::mtp
