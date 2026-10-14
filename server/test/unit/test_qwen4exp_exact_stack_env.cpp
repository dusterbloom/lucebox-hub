// Pure checks of the exact-stack default policy (qwen4exp_loader.cpp): which
// quant profiles turn the 38.41 decode fusion set on, and that the master switch
// and explicitly set knobs win over the defaults. No model, no GPU.
#include "qwen4exp/qwen4exp_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using luce::common::Qwen4ExpTensorType;
using luce::common::apply_qwen4exp_exact_stack_env;
using luce::common::qwen4exp_exact_stack_knobs;
using luce::common::qwen4exp_exact_stack_mismatch;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

// The shipped UD-Q4_K_XL profile: 48 layers, every 4th a full-attention layer,
// MoE in every layer, routed down Q8_0 in layers 2/4/30/46/47.
static std::vector<Qwen4ExpTensorType> shipped(int n_layer = 48, int interval = 4) {
    std::vector<Qwen4ExpTensorType> t;
    auto add = [&](int il, const char * suffix, ggml_type type) {
        t.push_back({"blk." + std::to_string(il) + "." + suffix, type});
    };
    for (int il = 0; il < n_layer; ++il) {
        if ((il + 1) % interval == 0) {
            add(il, "attn_q.weight", GGML_TYPE_Q8_0);
            add(il, "attn_k.weight", GGML_TYPE_Q8_0);
            add(il, "attn_v.weight", GGML_TYPE_Q8_0);
            add(il, "attn_output.weight", GGML_TYPE_Q8_0);
            add(il, "indexer.q_proj.weight", GGML_TYPE_BF16);
        } else {
            add(il, "attn_qkv.weight", GGML_TYPE_Q8_0);
            add(il, "attn_gate.weight", GGML_TYPE_Q8_0);
            add(il, "ssm_out.weight", GGML_TYPE_Q8_0);
            add(il, "ssm_alpha.weight", GGML_TYPE_F32);
        }
        for (const char * hc : {"hc_attn_up.weight", "hc_attn_down.weight", "hc_ffn_up.weight", "hc_ffn_down.weight"}) {
            add(il, hc, GGML_TYPE_Q8_0);
        }
        add(il, "ffn_gate_inp.weight", GGML_TYPE_F32);
        add(il, "ffn_gate_shexp.weight", GGML_TYPE_Q8_0);
        add(il, "ffn_up_shexp.weight", GGML_TYPE_Q8_0);
        add(il, "ffn_down_shexp.weight", GGML_TYPE_Q8_0);
        add(il, "ffn_gate_exps.weight", GGML_TYPE_Q4_K);
        add(il, "ffn_up_exps.weight", GGML_TYPE_Q4_K);
        const bool q8_down = il == 2 || il == 4 || il == 30 || il == 46 || il == 47;
        add(il, "ffn_down_exps.weight", q8_down ? GGML_TYPE_Q8_0 : GGML_TYPE_Q5_1);
    }
    t.push_back({"output.weight", GGML_TYPE_Q8_0});
    t.push_back({"token_embd.weight", GGML_TYPE_Q8_0});
    t.push_back({"per_layer_token_embd.weight", GGML_TYPE_IQ4_NL});
    return t;
}

static void retype(std::vector<Qwen4ExpTensorType> & t, const char * suffix, ggml_type type) {
    for (Qwen4ExpTensorType & x : t) {
        if (x.name.size() > std::strlen(suffix) &&
            x.name.compare(x.name.size() - std::strlen(suffix), std::string::npos, suffix) == 0) {
            x.type = type;
        }
    }
}

static void clear_knobs() {
    unsetenv("LUCE_QWEN_EXACT_STACK");
    for (const auto & knob : qwen4exp_exact_stack_knobs()) unsetenv(knob.first);
}

int main() {
    // The shipped profile and the validated A3/A4 requants (dense trunk and HC
    // mixers at Q6_K, shared expert kept Q8_0) match.
    CHECK(qwen4exp_exact_stack_mismatch(shipped(), 48).empty());
    {
        std::vector<Qwen4ExpTensorType> a4 = shipped();
        for (const char * s : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "attn_q.weight",
                               "attn_k.weight", "attn_v.weight", "attn_output.weight", "hc_attn_up.weight",
                               "hc_attn_down.weight", "hc_ffn_up.weight", "hc_ffn_down.weight"}) {
            retype(a4, s, GGML_TYPE_Q6_K);
        }
        CHECK(qwen4exp_exact_stack_mismatch(a4, 48).empty());
    }
    // A2 (shared expert requantized) crashed the HC up-mix contract: rejected.
    {
        std::vector<Qwen4ExpTensorType> a2 = shipped();
        retype(a2, "ffn_up_shexp.weight", GGML_TYPE_Q6_K);
        const std::string m = qwen4exp_exact_stack_mismatch(a2, 48);
        CHECK(m.find("ffn_up_shexp") != std::string::npos && m.find("q6_K") != std::string::npos);
    }
    // Other quant profiles of the same tensors are rejected with the tensor named.
    {
        std::vector<Qwen4ExpTensorType> t = shipped();
        retype(t, "hc_ffn_down.weight", GGML_TYPE_Q4_K);
        CHECK(qwen4exp_exact_stack_mismatch(t, 48).find("blk.0.hc_ffn_down.weight") == 0);
        t = shipped();
        retype(t, "ffn_up_exps.weight", GGML_TYPE_IQ4_XS);
        CHECK(!qwen4exp_exact_stack_mismatch(t, 48).empty());
        t = shipped();
        retype(t, "ffn_down_exps.weight", GGML_TYPE_Q4_K);
        CHECK(!qwen4exp_exact_stack_mismatch(t, 48).empty());
        t = shipped();
        retype(t, "ssm_out.weight", GGML_TYPE_Q5_1);
        CHECK(!qwen4exp_exact_stack_mismatch(t, 48).empty());
        t = shipped();
        retype(t, "attn_q.weight", GGML_TYPE_Q4_K);
        CHECK(!qwen4exp_exact_stack_mismatch(t, 48).empty());
    }
    // Routed gate/up Q5_K is inside the validated set (one such layer ships).
    {
        std::vector<Qwen4ExpTensorType> u = shipped();
        for (Qwen4ExpTensorType & x : u) if (x.name == "blk.7.ffn_gate_exps.weight") x.type = GGML_TYPE_Q5_K;
        CHECK(qwen4exp_exact_stack_mismatch(u, 48).empty());
    }
    // The contracts hard-code 48 layers as 36 GDN + 12 attention.
    CHECK(qwen4exp_exact_stack_mismatch(shipped(47), 47).find("block_count") == 0);
    CHECK(qwen4exp_exact_stack_mismatch(shipped(48, 6), 48).find("layer mix") == 0);
    // An MTP sidecar layer (blk.<n_layer>) is outside the contracts and ignored.
    {
        std::vector<Qwen4ExpTensorType> t = shipped();
        t.push_back({"blk.48.attn_q.weight", GGML_TYPE_Q4_K});
        t.push_back({"blk.48.attn_output.weight", GGML_TYPE_Q4_K});
        CHECK(qwen4exp_exact_stack_mismatch(t, 48).empty());
    }
    // The knob set is the full 38.41 env.
    {
        const auto & knobs = qwen4exp_exact_stack_knobs();
        CHECK(knobs.size() == 14);
        bool sh = false, warps = false;
        for (const auto & k : knobs) {
            if (!std::strcmp(k.first, "LUCE_QWEN_GRAPH_SH")) sh = !std::strcmp(k.second, "1");
            if (!std::strcmp(k.first, "LUCE_QWEN_EXPERT_ROW_WARPS")) warps = !std::strcmp(k.second, "8");
        }
        CHECK(sh && warps);
    }
    // Application: skipped off gfx1151, in reference mode, on a mismatch, and
    // under the master switch; otherwise every unset knob is set and an explicit
    // value is left alone.
    clear_knobs();
    CHECK(apply_qwen4exp_exact_stack_env(false, false, "") == -1);
    CHECK(apply_qwen4exp_exact_stack_env(true, true, "") == -1);
    CHECK(apply_qwen4exp_exact_stack_env(true, false, "blk.0.ffn_up_shexp.weight is q6_K") == -1);
    setenv("LUCE_QWEN_EXACT_STACK", "0", 1);
    CHECK(apply_qwen4exp_exact_stack_env(true, false, "") == -1);
    CHECK(std::getenv("LUCE_QWEN_GRAPH_SH") == nullptr);
    clear_knobs();
    setenv("LUCE_QWEN_EXPERT_ROW_WARPS", "4", 1);
    CHECK(apply_qwen4exp_exact_stack_env(true, false, "") == 13);
    CHECK(std::getenv("LUCE_QWEN_GRAPH_SH") && !std::strcmp(std::getenv("LUCE_QWEN_GRAPH_SH"), "1"));
    CHECK(std::getenv("LUCE_QWEN_EXPERT_ROW_WARPS") && !std::strcmp(std::getenv("LUCE_QWEN_EXPERT_ROW_WARPS"), "4"));
    CHECK(std::getenv("LUCE_GPU_ARGMAX") && !std::strcmp(std::getenv("LUCE_GPU_ARGMAX"), "1"));
    // A second application finds everything preset and sets nothing.
    CHECK(apply_qwen4exp_exact_stack_env(true, false, "") == 0);
    clear_knobs();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_qwen4exp_exact_stack_env: ok\n");
    return 0;
}
