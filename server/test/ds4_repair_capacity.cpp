// Repair-capacity ablation for the Qwen->DS4 bridge spike.
//
// The bridge predicts an approximate DS4 checkpoint at a cut and relies on the
// native suffix repair to make it usable. This tool measures how much the
// checkpoint can be degraded before repair stops compensating.
//
// Per (repair length R, degradation mode):
//   1. native prefill with a checkpoint at cut = L - R (slot 0);
//   2. degrade tensors in that checkpoint in place;
//   3. restore + native suffix repair + greedy decode, exactly the bridge's
//      PREFIX_REPAIR_R protocol;
//   4. compare the continuation against the undegraded protocol.
//
// Model-gated, not registered in ordinary CI.
//
//   ds4_repair_capacity MODEL.gguf PROMPT.txt|TOKENS.bin OUT.log [GPU]
//
// Env:
//   DS4_RC_CUTS   repair lengths in tokens, default "0,32,128,512"
//   DS4_RC_MODES  default "comp_zero,comp_tail,raw_zero,idx_zero,cs_zero,noise"
//   DS4_RC_NGEN   decoded tokens compared, default 32
//   DS4_RC_NOISE  noise sigma as a fraction of each row's std, default 0.1
//   DS4_RC_STRICT 1 = treat any histogram-visible failure as a nonzero exit

#include "deepseek4/deepseek4_backend.h"

#include "lbsnap_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

struct Case {
    int cut = 0;
    int R = 0;
    std::string mode;
    int matches = 0;
    int first_divergence = -1;
    std::vector<int32_t> tokens;
};

bool has_prefix(const char * name, const char * prefix) {
    return name && std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

std::vector<float> read_floats(ggml_tensor * t) {
    const size_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(raw[i]);
    } else {
        out.clear();
    }
    return out;
}

void write_floats(ggml_tensor * t, const std::vector<float> & v) {
    if (v.empty()) return;
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(v.size());
        for (size_t i = 0; i < v.size(); ++i) raw[i] = ggml_fp32_to_fp16(v[i]);
        ggml_backend_tensor_set(t, raw.data(), 0, raw.size() * sizeof(ggml_fp16_t));
    }
}

void zero_tensor(ggml_tensor * t) {
    std::vector<uint8_t> zeros(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
}

// Zero every row except the last along ne[1] (the snapshot row axis).
void zero_except_last_row(ggml_tensor * t) {
    const size_t row_bytes = ggml_type_size(t->type) * (size_t) std::max<int64_t>(t->ne[0], 0);
    const size_t total = ggml_nbytes(t);
    if (row_bytes == 0 || total <= row_bytes) return;
    std::vector<uint8_t> zeros(total - row_bytes, 0);
    ggml_backend_tensor_set(t, zeros.data(), 0, zeros.size());
}

void add_row_noise(ggml_tensor * t, float sigma_frac, std::mt19937_64 & rng) {
    std::vector<float> v = read_floats(t);
    if (v.empty()) return;
    const size_t row = (size_t) std::max<int64_t>(t->ne[0], 1);
    const size_t rows = v.size() / row;
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        for (size_t i = 0; i < row; ++i) mean += v[r * row + i];
        mean /= (float) row;
        float var = 0.0f;
        for (size_t i = 0; i < row; ++i) {
            const float d = v[r * row + i] - mean;
            var += d * d;
        }
        const float sigma = std::sqrt(var / (float) row);
        if (sigma <= 0.0f) continue;  // unused row
        for (size_t i = 0; i < row; ++i) {
            v[r * row + i] += dist(rng) * sigma_frac * sigma;
        }
    }
    write_floats(t, v);
}

int degrade(ggml_context * ctx, const std::string & mode, float noise_frac,
            std::mt19937_64 & rng) {
    if (!ctx || mode == "none") return 0;
    int touched = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
         t = ggml_get_next_tensor(ctx, t)) {
        const char * name = ggml_get_name(t);
        const std::string n = name ? name : "";
        if (mode == "comp_zero" && has_prefix(name, "ds4_snap_comp_kv_")) {
            zero_tensor(t); ++touched;
        } else if (mode == "comp_tail" && has_prefix(name, "ds4_snap_comp_kv_")) {
            zero_except_last_row(t); ++touched;
        } else if (mode == "raw_zero" && has_prefix(name, "ds4_snap_raw_kv_")) {
            zero_tensor(t); ++touched;
        } else if (mode == "idx_zero" && has_prefix(name, "ds4_snap_index_kv_")) {
            zero_tensor(t); ++touched;
        } else if (mode == "cs_zero" &&
                   (has_prefix(name, "ds4_snap_attn_cs_") ||
                    has_prefix(name, "ds4_snap_idx_cs_"))) {
            zero_tensor(t); ++touched;
        } else if (mode == "logits_zero" && n == "ds4_snap_last_logits") {
            zero_tensor(t); ++touched;
        } else if (mode == "noise" && has_prefix(name, "ds4_snap_comp_kv_")) {
            add_row_noise(t, noise_frac, rng); ++touched;
        }
    }
    return touched;
}

std::vector<Case> run_side(DeepSeek4Backend & backend,
                           const std::vector<int32_t> & prompt, int length,
                           const std::vector<int> & cuts,
                           const std::vector<std::string> & modes,
                           int n_gen, float noise_frac) {
    std::vector<Case> out;
    DaemonIO io;
    for (int cut : cuts) {
        for (const auto & mode : modes) {
            GenerateRequest a;
            a.prompt = prompt;
            a.n_gen = 0;
            a.snap_slot = 0;
            a.snap_pos = cut;
            if (!backend.generate(a, io).ok() || !backend.snapshot_used(0)) {
                std::fprintf(stderr, "FAIL: checkpoint at cut=%d (%s)\n", cut,
                             mode.c_str());
                continue;
            }
            std::mt19937_64 rng(1234);
            const ModelBackend::SnapshotRef ref = backend.snapshot_ref(0);
            const int touched = degrade(ref.ctx, mode, noise_frac, rng);

            GenerateRequest b;
            b.prompt = prompt;
            b.n_gen = n_gen;
            std::vector<int32_t> tokens;
            DaemonIO io2;
            io2.on_token = [&](int32_t tok) { tokens.push_back(tok); return true; };
            GenerateResult res = backend.restore_and_generate(0, b, io2);
            backend.snapshot_free(0);

            Case c;
            c.cut = cut;
            c.R = length - cut;
            c.mode = mode;
            c.tokens = res.ok() ? res.tokens : std::move(tokens);
            if (c.tokens.empty() && !res.ok()) c.tokens.assign(n_gen, -1);
            out.push_back(std::move(c));
            std::fprintf(stderr,
                         "[rc] R=%-5d mode=%-12s touched=%-3d decoded=%zu\n",
                         c.R, mode.c_str(), touched, out.back().tokens.size());
        }
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf PROMPT.txt|TOKENS.bin OUT.log [GPU]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const std::string out_path = argv[3];
    const int gpu = argc > 4 ? std::atoi(argv[4]) : 0;

    const std::vector<int32_t> prompt = lbsnap::read_prompt(argv[2], model);
    if (prompt.empty()) {
        std::fprintf(stderr, "FAIL: no tokens from %s\n", argv[2]);
        return 1;
    }
    const int length = (int) prompt.size();

    auto env_list = [](const char * key, const std::string & def) {
        const char * v = std::getenv(key);
        std::string s = v && v[0] ? v : def;
        std::vector<std::string> out;
        size_t p = 0;
        while (p <= s.size()) {
            const size_t c = s.find(',', p);
            const std::string item = s.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!item.empty()) out.push_back(item);
            if (c == std::string::npos) break;
            p = c + 1;
        }
        return out;
    };
    const std::string ng = std::getenv("DS4_RC_NGEN") ? std::getenv("DS4_RC_NGEN") : "32";
    const int n_gen = std::atoi(ng.c_str());
    const std::string nf = std::getenv("DS4_RC_NOISE") ? std::getenv("DS4_RC_NOISE") : "0.1";
    const float noise_frac = (float) std::atof(nf.c_str());

    std::vector<int> cuts;
    for (const auto & r : env_list("DS4_RC_CUTS", "0,32,128,512")) {
        const int R = std::atoi(r.c_str());
        cuts.push_back(std::max(0, length - R));
    }
    std::vector<std::string> modes = env_list(
        "DS4_RC_MODES", "none,comp_zero,comp_tail,raw_zero,idx_zero,cs_zero,noise");
    if (std::find(modes.begin(), modes.end(), "none") == modes.end()) {
        modes.insert(modes.begin(), "none");
    }

    DeepSeek4BackendConfig config;
    config.model_path = model;
#if defined(DFLASH27B_BACKEND_HIP)
    config.device.backend = PlacementBackend::Hip;
#else
    config.device.backend = PlacementBackend::Cuda;
#endif
    config.device.gpu = gpu;
    config.paged_attention = false;
    config.max_concurrency = 1;
    config.prefill_mode = PrefillAttentionMode::Exact;
    config.chunk = 512;
    config.max_ctx = length + 256;

    DeepSeek4Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }

    std::fprintf(stderr, "[rc] prompt=%d tokens, n_gen=%d, cuts(R)=%zu, modes=%zu\n",
                 length, n_gen, cuts.size(), modes.size());
    std::vector<Case> cases =
        run_side(backend, prompt, length, cuts, modes, n_gen, noise_frac);

    FILE * f = std::fopen(out_path.c_str(), "w");
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(f, "# repair-capacity ablation: prompt=%d tokens n_gen=%d "
                    "noise_frac=%.3f\n", length, n_gen, noise_frac);
    std::fprintf(f, "# R\tmode\tmatches\tfirst_divergence\n");

    int failures = 0;
    for (int cut : cuts) {
        const Case * ref = nullptr;
        for (const auto & c : cases) {
            if (c.cut == cut && c.mode == "none") ref = &c;
        }
        for (const auto & c : cases) {
            if (c.cut != cut) continue;
            Case & m = const_cast<Case &>(c);
            if (ref && &c != ref) {
                const size_t n = std::min(ref->tokens.size(), c.tokens.size());
                m.matches = 0;
                m.first_divergence = -1;
                for (size_t i = 0; i < n; ++i) {
                    if (ref->tokens[i] == c.tokens[i]) {
                        ++m.matches;
                    } else {
                        m.first_divergence = (int) i;
                        break;
                    }
                }
                if (m.first_divergence < 0 && c.tokens.size() == ref->tokens.size()) {
                    m.matches = (int) n;
                }
            } else {
                m.matches = (int) c.tokens.size();
            }
            std::fprintf(f, "%d\t%s\t%d\t%d\n", c.R, c.mode.c_str(), m.matches,
                         m.first_divergence);
            if (c.mode != "none" && m.matches < n_gen) ++failures;
            std::fprintf(stderr, "[rc] R=%-5d %-12s matches=%d/%d first_div=%d\n",
                         c.R, c.mode.c_str(), m.matches, n_gen, m.first_divergence);
        }
    }
    std::fclose(f);
    std::fprintf(stderr, "[rc] wrote %s (%d degraded cases lost tokens)\n",
                 out_path.c_str(), failures);
    return 0;
}
