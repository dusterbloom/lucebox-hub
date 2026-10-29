// Shared plumbing for the DS4.1 bridge research binaries (test_ds41_encoder_replay,
// ds41_bridge_probe): the lucebox6 launch profile as an in-process backend
// config, int32 token-id files, snapshot ring tampering, and a teacher-forced
// scorer that returns the full-vocab logits of every forced step.
//
// Model-gated; not registered in ordinary CI.
#pragma once

#include "common/peer_access.h"
#include "common/platform_env.h"
#include "deepseek4/deepseek4_backend.h"
#include "server/launch_profiles.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace ds41 {

using namespace luce::common;

using Logits = std::vector<std::vector<float>>;  // [step][vocab]

// Binary little-endian int32 ids, as written by the HF tokenization scripts.
inline std::vector<int32_t> read_ids(const std::string & path) {
    std::vector<int32_t> ids;
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return ids;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    ids.resize((size_t) bytes / sizeof(int32_t));
    if (std::fread(ids.data(), sizeof(int32_t), ids.size(), f) != ids.size()) ids.clear();
    std::fclose(f);
    return ids;
}

// The ds41-lucebox launch profile (R9700 target, Strix Halo expert owner) as
// environment + backend config, the way server_main applies it. Variables that
// are already set keep their value, as with the server. The sampler-visible
// settings that would make logits non-deterministic stay off: no drafter.
inline DeepSeek4BackendConfig make_config(const std::string & model, int max_ctx,
                                          const std::string & share_dir,
                                          const char * profile_name = "ds41-lucebox") {
    const luce::server::LaunchProfile * profile = luce::server::find_launch_profile(profile_name);
    if (!profile) {
        std::fprintf(stderr, "FAIL: no launch profile %s\n", profile_name);
        std::exit(2);
    }
    for (const auto & e : profile->env) {
        if (!std::getenv(e.name)) set_environment_variable(e.name, e.value, false);
    }
    // --expert-device hip:1 spelled as environment (server_main apply_expert_device).
    set_environment_variable("LUCE_DS4_MOE_TP", "1", false);
    set_environment_variable("LUCE_DS4_MOE_TP_INPROC", "1", false);
    set_environment_variable("LUCE_DS4_MOE_TP_GPU", "1", false);
    set_environment_variable("LUCE_DS4_MOE_TP_BACKEND", "hip", false);

    DeepSeek4BackendConfig cfg;
    cfg.model_path = model;
    cfg.device.backend = PlacementBackend::Hip;
    cfg.device.gpu = 0;
    cfg.device.peer_access = true;
    cfg.device.max_ctx = max_ctx;
    g_peer_access_opt_in = true;
    cfg.max_ctx = max_ctx;
    cfg.max_concurrency = 1;
    cfg.paged_attention = false;
    for (const auto & fl : profile->flags) {
        const std::string flag = fl.flag;
        const std::string value = fl.value ? fl.value : "";
        const auto path = [&] {
            return value.rfind("share/", 0) == 0 ? share_dir + "/" + value.substr(6) : value;
        };
        if (flag == "--chunk") cfg.chunk = std::atoi(value.c_str());
        else if (flag == "--ds4-prefill") {
            cfg.prefill_mode = value == "sparse" ? PrefillAttentionMode::Sparse
                             : value == "dense"  ? PrefillAttentionMode::Dense
                                                 : PrefillAttentionMode::Exact;
        }
        else if (flag == "--ds4-expert-placement") cfg.expert_placement_path = path();
        else if (flag == "--ds4-router-bias") cfg.router_bias_path = path();
        else if (flag == "--ds4-protected-experts") cfg.protected_experts_path = path();
    }
    return cfg;
}

// ── snapshot ring tampering ─────────────────────────────────────────────

inline bool is_ring(const char * name) {
    return name && std::strncmp(name, "ds4_snap_raw_kv_", 16) == 0;
}

inline std::vector<ggml_tensor *> snapshot_tensors(ggml_context * ctx, const char * prefix) {
    std::vector<ggml_tensor *> out;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        const char * n = ggml_get_name(t);
        if (n && std::strncmp(n, prefix, std::strlen(prefix)) == 0) out.push_back(t);
    }
    return out;
}

inline std::vector<uint8_t> tensor_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> b(ggml_nbytes(t));
    ggml_backend_tensor_get(t, b.data(), 0, b.size());
    return b;
}

inline void set_tensor_bytes(ggml_tensor * t, const std::vector<uint8_t> & b) {
    ggml_backend_tensor_set(t, b.data(), 0, b.size());
}

inline void zero_tensor(ggml_tensor * t) {
    set_tensor_bytes(t, std::vector<uint8_t>(ggml_nbytes(t), 0));
}

// N(0,1) values in the tensor's own element type (F16 or F32 only).
inline bool randomize_tensor(ggml_tensor * t, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    const size_t n = (size_t) ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> v(n);
        for (auto & x : v) x = dist(rng);
        ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> v(n);
        for (auto & x : v) x = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(ggml_fp16_t));
    } else {
        return false;
    }
    return true;
}

// ── teacher-forced scoring ──────────────────────────────────────────────

// Runs `run` (a generate / restore_and_generate / generate_in_place call that
// must request n_gen == forced.size()) with the decode oracle feeding `forced`,
// and returns the logits that predicted each forced token: step 0 is the
// prefill's last logits, step i the decode forward of forced[i-1].
template <typename Run>
inline bool teacher_force(DeepSeek4Backend & backend, const std::vector<int32_t> & forced,
                          Logits & out, Run && run) {
    out.clear();
    backend.set_decode_oracle([&](int step, const float * logits, int n_vocab) {
        out.emplace_back(logits, logits + n_vocab);
        return forced[(size_t) step];
    });
    GenerateResult res = run();
    backend.set_decode_oracle(nullptr);
    if (!res.ok() || out.size() != forced.size()) {
        std::fprintf(stderr, "FAIL: teacher-forced run: ok=%d steps=%zu/%zu\n",
                     res.ok() ? 1 : 0, out.size(), forced.size());
        return false;
    }
    return true;
}

// Largest |a-b| over two logit sets (INFINITY on a shape mismatch).
inline double max_abs_diff(const Logits & a, const Logits & b) {
    if (a.size() != b.size()) return INFINITY;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return INFINITY;
        for (size_t j = 0; j < a[i].size(); ++j) {
            const double d = std::fabs((double) a[i][j] - (double) b[i][j]);
            if (!(d <= worst)) worst = std::isnan(d) ? INFINITY : d;
        }
    }
    return worst;
}

inline bool bit_identical(const Logits & a, const Logits & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size() ||
            std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) != 0) return false;
    }
    return true;
}

}  // namespace ds41
