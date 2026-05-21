// test_cross_arch_drafter_loader.cpp
//
// TDD red->green for cross-arch GGUF loader adapter.
// Tests that SmolLM2 (llama-arch, no QK-norm) loads correctly via the adapted
// load_qwen3_drafter_model() path.
//
// Requires GPU. GGUF paths injected via env:
//   SMOLLM2_360M_PATH  -- path to SmolLM2-360M-BF16.gguf
//   SMOLLM2_135M_PATH  -- path to SmolLM2-135M-BF16.gguf
//
// If paths are absent the tests are skipped (exit 0) so CI without model
// files does not fail.

#include "qwen3/qwen3_drafter_model.h"
#include "internal.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace dflash::common;

static int n_pass = 0, n_fail = 0, n_skip = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (msg)); \
            ++n_fail; \
        } else { \
            std::fprintf(stderr, "PASS  %s\n", (msg)); \
            ++n_pass; \
        } \
    } while (0)

#define SKIP(msg) \
    do { \
        std::fprintf(stderr, "SKIP  %s\n", (msg)); \
        ++n_skip; \
    } while (0)

static void test_load_smollm2_360m(ggml_backend_t backend) {
    const char * path = std::getenv("SMOLLM2_360M_PATH");
    if (!path) { SKIP("SMOLLM2_360M_PATH not set"); return; }

    Qwen3DrafterWeights w;
    bool ok = load_qwen3_drafter_model(path, backend, w);
    CHECK(ok, "SmolLM2-360M: load succeeds");
    if (!ok) {
        std::fprintf(stderr, "  loader error: %s\n", dflash27b_last_error());
        return;
    }

    // SmolLM2-360M: 32 layers, head_dim=64, n_vocab=49152, rope_theta=100000
    CHECK(w.n_layer == 32,   "SmolLM2-360M: n_layer == 32");
    CHECK(w.n_embd  == 960,  "SmolLM2-360M: n_embd == 960");
    CHECK(w.n_head  == 15,   "SmolLM2-360M: n_head == 15");
    CHECK(w.n_head_kv == 5,  "SmolLM2-360M: n_head_kv == 5");
    CHECK(w.head_dim == 64,  "SmolLM2-360M: head_dim == 64");
    CHECK(w.n_vocab == 49152, "SmolLM2-360M: n_vocab == 49152");

    // QK-norm tensors must be nullptr for llama-arch (no per-head QK-norm)
    CHECK(w.layers[0].q_norm == nullptr, "SmolLM2-360M: layer[0].q_norm is nullptr");
    CHECK(w.layers[0].k_norm == nullptr, "SmolLM2-360M: layer[0].k_norm is nullptr");

    // Top-level tensors must be allocated
    CHECK(w.tok_embd != nullptr, "SmolLM2-360M: tok_embd allocated");
    CHECK(w.out_norm != nullptr, "SmolLM2-360M: out_norm allocated");
    CHECK(w.output   != nullptr, "SmolLM2-360M: output allocated");

    free_qwen3_drafter_model(w);
    CHECK(w.ctx == nullptr, "SmolLM2-360M: ctx freed");
}

static void test_load_smollm2_135m(ggml_backend_t backend) {
    const char * path = std::getenv("SMOLLM2_135M_PATH");
    if (!path) { SKIP("SMOLLM2_135M_PATH not set"); return; }

    Qwen3DrafterWeights w;
    bool ok = load_qwen3_drafter_model(path, backend, w);
    CHECK(ok, "SmolLM2-135M: load succeeds");
    if (!ok) {
        std::fprintf(stderr, "  loader error: %s\n", dflash27b_last_error());
        return;
    }

    // SmolLM2-135M: 30 layers, head_dim=64, n_vocab=49152
    CHECK(w.n_layer == 30,   "SmolLM2-135M: n_layer == 30");
    CHECK(w.n_embd  == 576,  "SmolLM2-135M: n_embd == 576");
    CHECK(w.n_head  == 9,    "SmolLM2-135M: n_head == 9");
    CHECK(w.n_head_kv == 3,  "SmolLM2-135M: n_head_kv == 3");
    CHECK(w.head_dim == 64,  "SmolLM2-135M: head_dim == 64");
    CHECK(w.n_vocab == 49152, "SmolLM2-135M: n_vocab == 49152");

    CHECK(w.layers[0].q_norm == nullptr, "SmolLM2-135M: layer[0].q_norm is nullptr");
    CHECK(w.layers[0].k_norm == nullptr, "SmolLM2-135M: layer[0].k_norm is nullptr");

    CHECK(w.tok_embd != nullptr, "SmolLM2-135M: tok_embd allocated");

    free_qwen3_drafter_model(w);
    CHECK(w.ctx == nullptr, "SmolLM2-135M: ctx freed");
}

static void test_qwen3_path_unchanged(ggml_backend_t backend) {
    // Verify Qwen3-0.6B still loads if path is provided.
    const char * path = std::getenv("QWEN3_06B_PATH");
    if (!path) { SKIP("QWEN3_06B_PATH not set -- qwen3 regression check skipped"); return; }

    Qwen3DrafterWeights w;
    bool ok = load_qwen3_drafter_model(path, backend, w);
    CHECK(ok, "Qwen3-0.6B: load still succeeds after adapter");
    if (!ok) {
        std::fprintf(stderr, "  loader error: %s\n", dflash27b_last_error());
        return;
    }
    CHECK(w.n_layer == 28,    "Qwen3-0.6B: n_layer == 28");
    CHECK(w.head_dim == 128,  "Qwen3-0.6B: head_dim == 128");
    CHECK(w.n_vocab == 151936, "Qwen3-0.6B: n_vocab == 151936");
    // Qwen3 has QK-norm
    CHECK(w.layers[0].q_norm != nullptr, "Qwen3-0.6B: layer[0].q_norm present");
    CHECK(w.layers[0].k_norm != nullptr, "Qwen3-0.6B: layer[0].k_norm present");
    free_qwen3_drafter_model(w);
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "SKIP  no CUDA backend available\n");
        return 0;
    }
    std::fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));

    test_load_smollm2_360m(backend);
    test_load_smollm2_135m(backend);
    test_qwen3_path_unchanged(backend);

    ggml_backend_free(backend);

    std::fprintf(stderr, "\n%d passed  %d failed  %d skipped\n", n_pass, n_fail, n_skip);
    return (n_fail > 0) ? 1 : 0;
}
