#include "kimi_k3/kimi_k3_backend.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

bool parse_ids(const char * text, std::vector<int32_t> & ids) {
    ids.clear();
    if (!text || !*text) return false;
    const char * cursor = text;
    while (*cursor) {
        errno = 0;
        char * end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (errno || end == cursor || value < 0 ||
            value > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        ids.push_back(static_cast<int32_t>(value));
        if (*end == '\0') break;
        if (*end != ',') return false;
        cursor = end + 1;
    }
    return !ids.empty();
}

nlohmann::json result_json(const KimiK3OracleVerifyResult & result) {
    const double committed_seconds =
        result.verify_seconds + result.commit_seconds;
    const double verify_speedup = result.verify_seconds > 0.0
        ? result.sequential_seconds / result.verify_seconds : 0.0;
    const double committed_speedup = committed_seconds > 0.0
        ? result.sequential_seconds / committed_seconds : 0.0;
    return {
        {"width", result.width},
        {"sequential_seconds", result.sequential_seconds},
        {"sequential_seconds_per_token",
         result.sequential_seconds / result.width},
        {"verify_seconds", result.verify_seconds},
        {"commit_seconds", result.commit_seconds},
        {"verify_plus_commit_seconds", committed_seconds},
        {"V_m_verify_only", verify_speedup},
        {"V_m_committed", committed_speedup},
        {"sequential_storage_bytes", result.sequential_storage_bytes},
        {"verify_storage_bytes", result.verify_storage_bytes},
        {"sequential_logical_provider_bytes",
         result.sequential_logical_provider_bytes},
        {"verify_logical_provider_bytes",
         result.verify_logical_provider_bytes},
        {"logical_provider_traffic_equal",
         result.sequential_logical_provider_bytes ==
             result.verify_logical_provider_bytes},
        {"sequential_compact_attempted",
         result.sequential_compact_attempted},
        {"sequential_compact_completed",
         result.sequential_compact_completed},
        {"sequential_compact_fallbacks",
         result.sequential_compact_fallbacks},
        {"sequential_compact_invalid", result.sequential_compact_invalid},
        {"verify_compact_attempted", result.verify_compact_attempted},
        {"verify_compact_completed", result.verify_compact_completed},
        {"verify_compact_fallbacks", result.verify_compact_fallbacks},
        {"verify_compact_invalid", result.verify_compact_invalid},
        {"logits_bit_equal", result.logits_bit_equal},
        {"argmax_bit_equal", result.argmax_bit_equal},
        {"logits_max_abs", result.logits_max_abs},
        {"logits_rel_l2", result.logits_rel_l2},
        {"recurrent_state_hash_equal",
         result.recurrent_state_hash_equal},
        {"mla_rows_hash_equal", result.mla_rows_hash_equal},
        {"sequential_recurrent_hash",
         result.sequential_recurrent_hash},
        {"verify_recurrent_hash", result.verify_recurrent_hash},
        {"sequential_mla_hash", result.sequential_mla_hash},
        {"verify_mla_hash", result.verify_mla_hash},
        {"first_hidden_mismatch_layer",
         result.first_hidden_mismatch_layer},
        {"first_hidden_mismatch_token",
         result.first_hidden_mismatch_token},
        {"first_hidden_max_abs", result.first_hidden_max_abs},
        {"first_hidden_rel_l2", result.first_hidden_rel_l2},
        {"first_conv_state_mismatch_layer",
         result.first_conv_state_mismatch_layer},
        {"first_ssm_state_mismatch_layer",
         result.first_ssm_state_mismatch_layer},
        {"first_mla_row_mismatch_layer",
         result.first_mla_row_mismatch_layer},
        {"sequential_conv_layer_hashes",
         result.sequential_conv_layer_hashes},
        {"verify_conv_layer_hashes", result.verify_conv_layer_hashes},
        {"sequential_ssm_layer_hashes",
         result.sequential_ssm_layer_hashes},
        {"verify_ssm_layer_hashes", result.verify_ssm_layer_hashes},
        {"sequential_mla_layer_hashes",
         result.sequential_mla_layer_hashes},
        {"verify_mla_layer_hashes", result.verify_mla_layer_hashes},
    };
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <kimi-k3.gguf> <prompt_ids_csv> <oracle_ids_csv> "
            "<result.json> [gpu=0] [core=cpu|accelerator] [expert_gpu=-1] "
            "[max_width=8] [min_width=2]\n",
            argv[0]);
        return 2;
    }
    std::vector<int32_t> prompt;
    std::vector<int32_t> oracle;
    if (!parse_ids(argv[2], prompt) || !parse_ids(argv[3], oracle) ||
        oracle.size() < 8) {
        std::fprintf(stderr,
            "[kimi-k3-s0] prompt IDs must be nonempty and oracle IDs must "
            "contain at least eight tokens\n");
        return 2;
    }

    KimiK3BackendConfig config;
    config.model_path = argv[1];
    config.device.gpu = argc > 5 ? std::atoi(argv[5]) : 0;
    config.device.max_ctx = 4096;
    config.oracle_verify_tokens = 8;
    const char * layer_diagnostics =
        std::getenv("DFLASH_KIMI_S0_LAYER_CAPTURE");
    config.oracle_layer_diagnostics = layer_diagnostics &&
        *layer_diagnostics && std::string(layer_diagnostics) != "0";
    config.moe_storage = MoeStoragePolicy::Ssd;
    if (argc > 6 && !parse_kimi_k3_core_placement(
            argv[6], config.core_placement)) {
        std::fprintf(stderr, "[kimi-k3-s0] core must be cpu or accelerator\n");
        return 2;
    }
    config.expert_gpu = argc > 7 ? std::atoi(argv[7]) : -1;
    const int max_width = argc > 8 ? std::atoi(argv[8]) : 8;
    const int min_width = argc > 9 ? std::atoi(argv[9]) : 2;
    if (max_width != 2 && max_width != 4 && max_width != 8) {
        std::fprintf(stderr, "[kimi-k3-s0] max_width must be 2, 4, or 8\n");
        return 2;
    }
    if ((min_width != 2 && min_width != 4 && min_width != 8) ||
        min_width > max_width) {
        std::fprintf(stderr,
            "[kimi-k3-s0] min_width must be 2, 4, or 8 and <= max_width\n");
        return 2;
    }

    KimiK3Backend backend(config);
    if (!backend.init()) return 1;

    nlohmann::json report = {
        {"experiment", "S0_ORACLE_KDA_VERIFY"},
        {"status", "MEASURED"},
        {"prompt_ids", prompt},
        {"oracle_ids", oracle},
        {"widths", nlohmann::json::array()},
        {"notes", {
            "Physical storage bytes are Linux /proc/self/io read_bytes deltas. "
            "Logical provider/H2D bytes remain sourced from the existing P20 "
            "traffic and I/O traces. AttnRes has no cross-token persistent "
            "cache; terminal logit parity covers its accepted-boundary result."
        }},
    };
    bool parity_failed = false;
    for (int width : {2, 4, 8}) {
        if (width < min_width) continue;
        if (width > max_width) break;
        const std::vector<int32_t> tokens(
            oracle.begin(), oracle.begin() + width);
        KimiK3OracleVerifyResult result;
        std::string error;
        if (!backend.benchmark_oracle_verify(
                prompt, tokens, result, &error)) {
            std::fprintf(stderr,
                "[kimi-k3-s0] width=%d failed: %s\n",
                width, error.c_str());
            report["status"] = "FAILED";
            report["failure"] = {{"width", width}, {"error", error}};
            std::ofstream output(argv[4]);
            output << report.dump(2) << '\n';
            return 1;
        }
        report["widths"].push_back(result_json(result));
        const bool parity = result.logits_bit_equal &&
            result.argmax_bit_equal && result.recurrent_state_hash_equal &&
            result.mla_rows_hash_equal;
        if (!parity) {
            parity_failed = true;
            report["status"] = "FAILED";
            report["parity_gate"] = {
                {"status", "FAIL"},
                {"width", width},
                {"reason", "sequential/block accepted-boundary mismatch"},
            };
        }
        std::fprintf(stderr,
            "[kimi-k3-s0] width=%d sequential=%.3fs verify=%.3fs "
            "commit=%.3fs V=%.3fx logits=%s recurrent=%s mla=%s\n",
            width, result.sequential_seconds, result.verify_seconds,
            result.commit_seconds,
            result.sequential_seconds /
                (result.verify_seconds + result.commit_seconds),
            result.logits_bit_equal ? "BIT_EQUAL" : "DIFF",
            result.recurrent_state_hash_equal ? "HASH_EQUAL" : "DIFF",
            result.mla_rows_hash_equal ? "HASH_EQUAL" : "DIFF");
        if (parity_failed) break;
    }
    if (!parity_failed) report["parity_gate"] = {{"status", "PASS"}};
    std::ofstream output(argv[4]);
    output << report.dump(2) << '\n';
    if (!output) {
        std::fprintf(stderr, "[kimi-k3-s0] cannot write %s\n", argv[4]);
        return 1;
    }
    return parity_failed ? 3 : 0;
}
