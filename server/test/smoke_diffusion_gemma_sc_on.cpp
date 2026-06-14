// Smoke test: run a single DiffusionGemma forward (SC-on) on the golden
// teacher-forced input and dump canvas logits to ours_sc_on.bin.
//
// Usage:
//   smoke_diffusion_gemma_sc_on <model.gguf> <thoughts_dir>
//
// Reads:  <thoughts_dir>/golden/teacher_forced/input.json
//         <thoughts_dir>/golden/teacher_forced/prev_logits.bin  [C_dump * n_vocab F32]
// Outputs: <thoughts_dir>/golden/teacher_forced/ours_sc_on.bin  [C_dump * n_vocab F32]

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

static float parse_json_float(const std::string & s, const std::string & key) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) throw std::runtime_error("key not found: " + key);
    auto c = s.find(':', p);
    std::istringstream ss(s.substr(c + 1));
    float v; ss >> v;
    return v;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

static std::vector<float> read_bin_f32(const std::string & path, size_t n_floats) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open binary: " + path);
    std::vector<float> buf(n_floats);
    f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize)(n_floats * sizeof(float)));
    if (!f) throw std::runtime_error("short read from: " + path);
    return buf;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <diffusiongemma.gguf> <thoughts_dir>\n",
            argv[0]);
        return 2;
    }
    const std::string model_path   = argv[1];
    const std::string thoughts_dir = argv[2];
    const std::string input_path   = thoughts_dir + "/golden/teacher_forced/input.json";
    const std::string prev_path    = thoughts_dir + "/golden/teacher_forced/prev_logits.bin";
    const std::string out_path     = thoughts_dir + "/golden/teacher_forced/ours_sc_on.bin";

    std::string json;
    try { json = read_file(input_path); }
    catch (const std::exception & e) {
        std::fprintf(stderr, "cannot read input.json: %s\n", e.what());
        return 1;
    }

    const int P        = parse_json_int(json, "P");
    const int C_dump   = parse_json_int(json, "C_dump");
    const int C_model  = parse_json_int(json, "C_model");
    const int n_vocab  = parse_json_int(json, "n_vocab");
    const float sc_temp_inv = parse_json_float(json, "temp_inv_step0");

    std::vector<int32_t> prompt_tokens    = parse_json_int_array(json, "prompt_tokens");
    std::vector<int32_t> canvas_tokens_s0 = parse_json_int_array(json, "canvas_tokens_step0_all");

    std::printf("[sc_on] P=%d C_model=%d C_dump=%d n_vocab=%d sc_temp_inv=%.4f\n",
                P, C_model, C_dump, n_vocab, sc_temp_inv);

    // Load prev_logits.bin — golden prev-step logits [C_dump, n_vocab]
    // Note: prev_logits.bin only has C_dump rows; we need C_model rows for set_sc.
    // Pad the rest with zeros.
    std::vector<float> prev_logits;
    try {
        prev_logits = read_bin_f32(prev_path, (size_t)C_dump * (size_t)n_vocab);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "cannot read prev_logits.bin: %s\n", e.what());
        return 1;
    }
    // Pad to C_model rows if needed
    prev_logits.resize((size_t)C_model * (size_t)n_vocab, 0.0f);

    std::printf("[sc_on] loaded prev_logits (%zu floats, padded to %d rows)\n",
                prev_logits.size(), C_model);

    std::vector<int32_t> full_tokens;
    full_tokens.insert(full_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
    full_tokens.insert(full_tokens.end(), canvas_tokens_s0.begin(), canvas_tokens_s0.end());

    const int n_total   = P + C_model;
    const int max_ctx   = (n_total + 255) & ~255;

    DiffusionGemmaConfig cfg;
    cfg.model_path = model_path.c_str();
    cfg.gpu        = 0;
    cfg.max_ctx    = max_ctx;

    DiffusionGemmaGraph graph(cfg);
    if (!graph.init()) {
        std::fprintf(stderr, "[sc_on] graph.init() failed\n");
        return 1;
    }
    std::printf("[sc_on] model loaded\n");

    int out_prefix = 0;
    if (!graph.prepare(prompt_tokens, out_prefix)) {
        std::fprintf(stderr, "[sc_on] prepare() failed\n");
        return 1;
    }

    // SC-on: set prev logits with sc_use=1.0
    graph.set_sc(prev_logits.data(), /*sc_use=*/1.0f, sc_temp_inv);

    std::vector<float> out_logits;
    std::printf("[sc_on] running SC-on forward (n_tokens=%d, sc_use=1.0)...\n", n_total);
    std::fflush(stdout);

    if (!graph.forward_block(full_tokens, /*block_begin=*/P, /*block_len=*/C_model,
                             /*bidirectional=*/true, out_logits)) {
        std::fprintf(stderr, "[sc_on] forward_block() failed\n");
        return 1;
    }

    std::printf("[sc_on] forward ok, logits.size=%zu (expected %zu)\n",
                out_logits.size(), (size_t)n_vocab * (size_t)C_model);

    if ((int)out_logits.size() < C_dump * n_vocab) {
        std::fprintf(stderr, "[sc_on] not enough logits: got %zu, need %d\n",
                     out_logits.size(), C_dump * n_vocab);
        return 1;
    }

    {
        std::FILE * fout = std::fopen(out_path.c_str(), "wb");
        if (!fout) {
            std::fprintf(stderr, "[sc_on] cannot write %s\n", out_path.c_str());
            return 1;
        }
        const size_t row_floats = (size_t)n_vocab;
        for (int c = 0; c < C_dump; ++c) {
            const float * row = out_logits.data() + (size_t)c * row_floats;
            if (std::fwrite(row, sizeof(float), row_floats, fout) != row_floats) {
                std::fprintf(stderr, "[sc_on] write error at row %d\n", c);
                std::fclose(fout);
                return 1;
            }
        }
        std::fclose(fout);
        std::printf("[sc_on] wrote %s (%d rows x %d floats)\n",
                    out_path.c_str(), C_dump, n_vocab);
    }

    {
        const int V = n_vocab;
        int best = 0;
        for (int v = 1; v < V; ++v) {
            if (out_logits[v] > out_logits[best]) best = v;
        }
        std::printf("[sc_on] argmax row0 = %d  (logit %.4f)\n", best, out_logits[best]);
    }

    std::printf("[sc_on] PASS\n");
    return 0;
}
