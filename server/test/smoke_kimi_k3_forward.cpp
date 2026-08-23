#include "kimi_k3/kimi_k3_backend.h"
#include "server/tokenizer.h"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace dflash::common;

static bool load_prompt_ids(const std::string & path,
                            std::vector<int32_t> & ids,
                            std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open token file: " + path;
        return false;
    }
    std::string contents((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    if (input.bad()) {
        error = "cannot read token file: " + path;
        return false;
    }
    for (char & ch : contents) {
        if (ch == ',') ch = ' ';
    }
    std::istringstream tokens(contents);
    std::string token;
    while (tokens >> token) {
        int32_t value = 0;
        const char * const begin = token.data();
        const char * const end = begin + token.size();
        const auto parsed = std::from_chars(begin, end, value);
        if (parsed.ec != std::errc() || parsed.ptr != end) {
            error = "invalid int32 token in: " + path;
            return false;
        }
        ids.push_back(value);
    }
    if (ids.empty()) {
        error = "token file contains no IDs: " + path;
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <kimi-k3.gguf> [gpu=0] [n_gen=16] "
            "[prompt|@token-file]\n",
            argv[0]);
        return 2;
    }
    const char * model = argv[1];
    const int gpu = argc > 2 ? std::atoi(argv[2]) : 0;
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 16;
    const std::string prompt = argc > 4
        ? argv[4]
        : "According to all known laws";

    KimiK3BackendConfig config;
    config.model_path = model;
    config.device.backend = PlacementBackend::Hip;
    config.device.gpu = gpu;
    config.device.max_ctx = 4096;
    if (const char * raw = std::getenv("DFLASH_KIMI_SMOKE_MAX_CTX")) {
        const int requested = std::atoi(raw);
        if (requested > 0) config.device.max_ctx = requested;
    }

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model)) {
        std::fprintf(stderr, "[kimi-k3-smoke] tokenizer load failed\n");
        return 1;
    }
    std::vector<int32_t> prompt_ids;
    const bool token_file = !prompt.empty() && prompt.front() == '@';
    if (token_file) {
        std::string error;
        if (!load_prompt_ids(prompt.substr(1), prompt_ids, error)) {
            std::fprintf(stderr, "[kimi-k3-smoke] %s\n", error.c_str());
            return 1;
        }
    } else {
        prompt_ids = tokenizer.encode(prompt);
    }
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "[kimi-k3-smoke] prompt tokenized to zero IDs\n");
        return 1;
    }
    const size_t max_ctx = static_cast<size_t>(config.device.max_ctx);
    if (n_gen < 0 || prompt_ids.size() > max_ctx ||
        static_cast<size_t>(n_gen) > max_ctx - prompt_ids.size()) {
        std::fprintf(stderr,
                     "[kimi-k3-smoke] prompt (%zu) plus n_gen (%d) exceeds "
                     "max_ctx=%d\n",
                     prompt_ids.size(), n_gen, config.device.max_ctx);
        return 1;
    }
    KimiK3Backend backend(config);
    if (!backend.init()) return 1;

    GenerateRequest request;
    request.prompt = prompt_ids;
    request.n_gen = n_gen;
    request.do_sample = false;
    const GenerateResult result = backend.generate(request, {});
    if (!result.ok()) {
        std::fprintf(stderr, "[kimi-k3-smoke] generation failed: %s (%s)\n",
                     std::string(result.error_code()).c_str(),
                     std::string(result.error_detail()).c_str());
        return 1;
    }

    std::printf("[kimi-k3-smoke] prompt_ids:");
    for (int32_t id : prompt_ids) std::printf(" %d", id);
    std::printf("\n[kimi-k3-smoke] output_ids:");
    for (int32_t id : result.tokens) std::printf(" %d", id);
    std::printf("\n[kimi-k3-smoke] text: %s%s\n",
                token_file ? "" : prompt.c_str(),
                tokenizer.decode(result.tokens).c_str());

    const double prefill_rate = result.prefill_s > 0.0
        ? static_cast<double>(prompt_ids.size()) / result.prefill_s : 0.0;
    const size_t transitions = result.tokens.empty()
        ? 0 : result.tokens.size() - 1;
    const double ar_rate = result.decode_s > 0.0
        ? static_cast<double>(transitions) / result.decode_s : 0.0;
    std::printf(
        "[kimi-k3-smoke] prefill=%.3fs (%.6f positions/s) "
        "decode=%.3fs (%zu transitions, %.6f true-AR/s)\n",
        result.prefill_s, prefill_rate, result.decode_s, transitions, ar_rate);
    return 0;
}
