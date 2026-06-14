// Smoke test: run a single DiffusionGemma forward (SC-off) on the golden
// teacher-forced input and dump canvas logits to ours_sc_off.bin.
//
// Usage:
//   smoke_diffusion_gemma_forward <model.gguf> <thoughts_dir>
//
// Outputs: <thoughts_dir>/teacher_forced/ours_sc_off.bin  [C_dump * n_vocab F32]
//          (same layout as golden logits_sc_off.bin: first C_dump canvas rows)

#include "diffusion_gemma.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dflash::common;

// ── Minimal JSON int-array parser ──────────────────────────────────────────
// Reads a JSON array of integers: [a, b, c, ...]  (single-line, no nesting)
static std::vector<int32_t> parse_json_int_array(const std::string & s,
                                                   const std::string & key) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) throw std::runtime_error("key not found: " + key);
    auto ob = s.find('[', p);
    auto cb = s.find(']', ob);
    if (ob == std::string::npos || cb == std::string::npos)
        throw std::runtime_error("array not found for: " + key);
    std::vector<int32_t> out;
    std::string sub = s.substr(ob + 1, cb - ob - 1);
    std::istringstream ss(sub);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        out.push_back((int32_t)std::stoi(tok));
    }
    return out;
}

static int parse_json_int(const std::string & s, const std::string & key) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) throw std::runtime_error("key not found: " + key);
    auto c = s.find(':', p);
    std::istringstream ss(s.substr(c + 1));
    int v; ss >> v;
    return v;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <diffusiongemma.gguf> <thoughts_dir>\n"
            "  e.g. %s /home/peppi/models/diffusiongemma-26b/diffusiongemma-26B-A4B-it-Q4_K_M.gguf\n"
            "       /home/peppi/Dev/lucebox-hub/thoughts/diffusiongemma\n",
            argv[0], argv[0]);
        return 2;
    }
    const std::string model_path   = argv[1];
    const std::string thoughts_dir = argv[2];
    const std::string input_path   = thoughts_dir + "/golden/teacher_forced/input.json";
    const std::string out_path     = thoughts_dir + "/golden/teacher_forced/ours_sc_off.bin";

    // ── Load golden input ──────────────────────────────────────────────
    std::string json;
    try {
        json = read_file(input_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "cannot read input.json: %s\n", e.what());
        return 1;
    }
    int P      = parse_json_int(json, "P");
    int C_dump = parse_json_int(json, "C_dump");
    int C_model = parse_json_int(json, "C_model");
    std::vector<int32_t> prompt_tokens    = parse_json_int_array(json, "prompt_tokens");
    std::vector<int32_t> canvas_tokens_s0 = parse_json_int_array(json, "canvas_tokens_step0_all");

    std::printf("[smoke] P=%d C_model=%d C_dump=%d prompt_tokens=%zu canvas_tokens=%zu\n",
                P, C_model, C_dump, prompt_tokens.size(), canvas_tokens_s0.size());
    assert((int)prompt_tokens.size() == P);
    assert((int)canvas_tokens_s0.size() == C_model);

    // Full token sequence: [prompt | canvas]
    std::vector<int32_t> full_tokens;
    full_tokens.insert(full_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
    full_tokens.insert(full_tokens.end(), canvas_tokens_s0.begin(), canvas_tokens_s0.end());

    // ── Init model ────────────────────────────────────────────────────
    // max_ctx must be at least (n_tokens+255)&~255 because build_gemma4_attn_block
    // pads the FA kv_len to a 256 boundary and creates a cache view of that size.
    const int n_total = P + C_model;
    const int max_ctx_min = ((n_total + 255) & ~255);  // 512 for n=273

    DiffusionGemmaConfig cfg;
    cfg.model_path = model_path.c_str();
    cfg.gpu        = 0;
    cfg.max_ctx    = max_ctx_min;

    DiffusionGemmaGraph graph(cfg);
    if (!graph.init()) {
        std::fprintf(stderr, "[smoke] graph.init() failed\n");
        return 1;
    }
    std::printf("[smoke] model loaded. vocab=%d\n", graph.vocab());

    // ── Prepare (records P) ──────────────────────────────────────────
    int out_prefix = 0;
    if (!graph.prepare(prompt_tokens, out_prefix)) {
        std::fprintf(stderr, "[smoke] prepare() failed\n");
        return 1;
    }
    std::printf("[smoke] prepare ok, prefix_len=%d\n", out_prefix);

    // ── SC-off forward ───────────────────────────────────────────────
    // sc_use=0: SC subgraph is well-formed but outputs zero → same as SC-off.
    // We don't call set_sc, so sc_logits_ptr_=nullptr → no SC path at all.
    std::vector<float> out_logits;
    std::printf("[smoke] running forward (SC-off, sc_use=0, n_tokens=%d)...\n",
                P + C_model);
    std::fflush(stdout);

    if (!graph.forward_block(full_tokens, /*block_begin=*/P, /*block_len=*/C_model,
                             /*bidirectional=*/true, out_logits)) {
        std::fprintf(stderr, "[smoke] forward_block() failed\n");
        return 1;
    }
    std::printf("[smoke] forward ok, logits.size=%zu (expected %zu)\n",
                out_logits.size(), (size_t)graph.vocab() * (size_t)C_model);

    if ((int)out_logits.size() < C_dump * graph.vocab()) {
        std::fprintf(stderr, "[smoke] not enough logits: got %zu, need %d\n",
                     out_logits.size(), C_dump * graph.vocab());
        return 1;
    }

    // ── Dump first C_dump canvas rows (same layout as reference golden) ──
    // out_logits layout: [n_vocab, C_model] row-major (canvas row 0 first).
    // Golden layout in logits_sc_off.json: shape=[16,262144] = [C_dump, n_vocab].
    // Both layouts are C_dump rows of n_vocab floats — compatible.
    {
        std::FILE * fout = std::fopen(out_path.c_str(), "wb");
        if (!fout) {
            std::fprintf(stderr, "[smoke] cannot write %s\n", out_path.c_str());
            return 1;
        }
        // Write first C_dump canvas rows (each row = n_vocab floats)
        const size_t row_floats = (size_t)graph.vocab();
        for (int c = 0; c < C_dump; ++c) {
            const float * row = out_logits.data() + (size_t)c * row_floats;
            if (std::fwrite(row, sizeof(float), row_floats, fout) != row_floats) {
                std::fprintf(stderr, "[smoke] write error at row %d\n", c);
                std::fclose(fout);
                return 1;
            }
        }
        std::fclose(fout);
        std::printf("[smoke] wrote %s (%d rows x %d vocab floats)\n",
                    out_path.c_str(), C_dump, graph.vocab());
    }

    // ── Quick argmax spot-check (row 0) ──────────────────────────────
    {
        const int V = graph.vocab();
        int best = 0;
        for (int v = 1; v < V; ++v) {
            if (out_logits[v] > out_logits[best]) best = v;
        }
        std::printf("[smoke] argmax row0 = %d  (logit %.4f)\n", best, out_logits[best]);
    }

    std::printf("[smoke] PASS\n");
    return 0;
}
