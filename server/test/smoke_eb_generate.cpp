// Smoke test: run the full entropy-bound decode loop on a real DiffusionGemma
// model for the 3 golden prompts (fib/sky/math). Validates G3 coherence.
//
// Usage:
//   smoke_eb_generate <model.gguf> <prompt.bin> [n_gen] [seed] [max_steps]
//
// <prompt.bin>: raw int32 token ids (little-endian), no header
// Outputs the raw argmax canvas token IDs and, if vocab can be read, text.
//
// Reads vocab strings from the GGUF file for detokenization (SentencePiece
// ▁-prefix decoding: strip leading ▁ and replace inner ▁ with space).

#include "diffusion_gemma.h"
#include "diffusion_decoder.h"
#include "diffusion_types.h"

#include "gguf.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dflash::common;

static std::vector<int32_t> read_i32_bin(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(std::string("cannot open: ") + path);
    const size_t sz = (size_t)f.tellg();
    if (sz % 4 != 0) throw std::runtime_error("file size not a multiple of 4");
    f.seekg(0);
    std::vector<int32_t> v(sz / 4);
    f.read(reinterpret_cast<char *>(v.data()), (std::streamsize)sz);
    return v;
}

// Load vocab strings from a GGUF file (SentencePiece token list).
static std::vector<std::string> load_vocab(const char * model_path) {
    struct gguf_init_params p = { .no_alloc = true, .ctx = nullptr };
    gguf_context * gc = gguf_init_from_file(model_path, p);
    if (!gc) return {};

    const int ki = gguf_find_key(gc, "tokenizer.ggml.tokens");
    if (ki < 0) { gguf_free(gc); return {}; }

    const int n = (int)gguf_get_arr_n(gc, ki);
    std::vector<std::string> vocab(n);
    for (int i = 0; i < n; ++i) {
        const char * s = gguf_get_arr_str(gc, ki, i);
        vocab[i] = s ? s : "";
    }
    gguf_free(gc);
    return vocab;
}

// SentencePiece decode: strip leading ▁ (U+2581), replace inner ▁ with space.
static std::string sp_decode(const std::vector<int32_t> & ids,
                              const std::vector<std::string> & vocab) {
    std::string out;
    for (const int32_t id : ids) {
        if (id < 0 || id >= (int)vocab.size()) { out += '?'; continue; }
        std::string piece = vocab[id];
        // Replace ▁ (UTF-8 E2 96 81) with space; strip leading one.
        const std::string sp_space = "\xe2\x96\x81";  // ▁
        if (piece.size() >= 3 && piece.substr(0, 3) == sp_space) {
            // leading ▁ → space (or nothing if at start)
            piece = (out.empty() ? "" : " ") + piece.substr(3);
        }
        // Replace remaining ▁ with space
        size_t pos = 0;
        while ((pos = piece.find(sp_space, pos)) != std::string::npos) {
            piece.replace(pos, sp_space.size(), " ");
            pos += 1;
        }
        out += piece;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <model.gguf> <prompt.bin> [n_gen=256] [seed=42] [max_steps=48] [schedule_steps=12]\n",
            argv[0]);
        return 2;
    }
    const char * model_path     = argv[1];
    const char * prompt_path    = argv[2];
    const int    n_gen          = argc > 3 ? std::atoi(argv[3]) : 256;
    const uint64_t seed         = argc > 4 ? (uint64_t)std::atoi(argv[4]) : 42ULL;
    const int    max_steps      = argc > 5 ? std::atoi(argv[5]) : 48;
    const int    schedule_steps = argc > 6 ? std::atoi(argv[6]) : 12;

    std::printf("[eb-gen] model=%s  prompt=%s  n_gen=%d  seed=%llu  max_steps=%d  schedule_steps=%d\n",
                model_path, prompt_path, n_gen, (unsigned long long)seed, max_steps, schedule_steps);
    std::fflush(stdout);

    // Load prompt
    std::vector<int32_t> prompt;
    try { prompt = read_i32_bin(prompt_path); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "[eb-gen] cannot load prompt: %s\n", e.what());
        return 1;
    }
    std::printf("[eb-gen] prompt: %d tokens\n", (int)prompt.size());

    // Load vocab for detokenization
    std::vector<std::string> vocab = load_vocab(model_path);
    std::printf("[eb-gen] vocab: %d tokens\n", (int)vocab.size());

    // Build model — max_ctx must be >= the attention min (sliding_window=1024 for SWA layers).
    // Use at least 1024 (SWA requirement) rounded up to the next multiple of 64.
    const int min_ctx = (int)prompt.size() + n_gen;
    const int max_ctx = ((std::max(min_ctx, 1024) + 63) / 64) * 64;
    DiffusionGemmaConfig gcfg;
    gcfg.model_path = model_path;
    gcfg.gpu        = 0;
    gcfg.max_ctx    = max_ctx;

    DiffusionGemmaGraph graph(gcfg);
    if (!graph.init()) {
        std::fprintf(stderr, "[eb-gen] model init failed\n");
        return 1;
    }
    std::printf("[eb-gen] model loaded\n");
    std::fflush(stdout);

    // Config: EntropyBound mode
    DiffusionConfig cfg;
    cfg.remasking              = DiffusionRemask::EntropyBound;
    cfg.noise_scheme           = DiffusionNoise::UniformState;
    cfg.block_size             = n_gen;         // single canvas block
    cfg.seed                   = seed;
    cfg.eb_max_steps           = max_steps;
    cfg.eb_schedule_steps      = schedule_steps;
    cfg.eb_t_min               = 0.4f;
    cfg.eb_t_max               = 0.8f;
    cfg.eb_entropy_bound       = 0.1f;
    cfg.eb_stability_threshold = 1;
    cfg.eb_confidence_threshold = 0.005f;

    SamplerCfg greedy{};   // temp=0 (unused by EB path, but required by signature)

    std::vector<int32_t> out_tokens;
    DiffusionStream stream;
    stream.on_token = [&out_tokens](int32_t tok) {
        out_tokens.push_back(tok);
        return true;
    };

    std::printf("[eb-gen] running EB decode (n_gen=%d, max_steps=%d, schedule_steps=%d)...\n",
                n_gen, max_steps, schedule_steps);
    std::fflush(stdout);

    DiffusionDecodeResult r = run_diffusion_generate(
        graph, prompt, n_gen, cfg, greedy, /*do_sample=*/false, stream);

    if (!r.ok) {
        std::fprintf(stderr, "[eb-gen] generation failed: %s\n", r.error.c_str());
        return 1;
    }

    std::printf("[eb-gen] done: %d tokens, %d forward passes, %d blocks\n",
                r.stats.tokens, r.stats.forward_passes, r.stats.blocks);

    // Detokenize
    if (!vocab.empty()) {
        std::string text = sp_decode(r.tokens, vocab);
        std::printf("\n=== GENERATED TEXT ===\n%s\n=== END ===\n", text.c_str());
    } else {
        std::printf("[eb-gen] no vocab, printing raw token IDs\n");
        for (int i = 0; i < (int)r.tokens.size(); ++i) {
            std::printf("%d", r.tokens[i]);
            if (i + 1 < (int)r.tokens.size()) std::printf(" ");
        }
        std::printf("\n");
    }

    return 0;
}
