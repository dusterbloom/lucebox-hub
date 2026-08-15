#include "kimi_k3/kimi_k3_backend.h"
#include "server/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <cstdlib>
#endif

using json = nlohmann::json;
using namespace dflash::common;

namespace {

struct SuiteEntry {
    std::string id;
    std::string split;
    std::string text;
    int model_layer = 0;
};

bool valid_id(const std::string & value) {
    return !value.empty() && std::all_of(
        value.begin(), value.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '-' ||
                character == '_';
        });
}

bool set_environment(const char * name, const std::string & value) {
#if defined(_WIN32)
    return _putenv_s(name, value.c_str()) == 0;
#else
    return setenv(name, value.c_str(), 1) == 0;
#endif
}

bool read_suite(const std::filesystem::path & path,
                std::vector<SuiteEntry> & entries,
                std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open suite " + path.string();
        return false;
    }
    std::set<std::string> identifiers;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        try {
            const json row = json::parse(line);
            if (!row.is_object() || !row.contains("id") ||
                !row.contains("split") || !row.contains("text") ||
                !row["id"].is_string() || !row["split"].is_string() ||
                !row["text"].is_string()) {
                error = "suite line " + std::to_string(line_number) +
                    " needs string id, split, and text fields";
                return false;
            }
            SuiteEntry entry{
                row["id"].get<std::string>(),
                row["split"].get<std::string>(),
                row["text"].get<std::string>(),
            };
            if (row.contains("model_layer")) {
                if (!row["model_layer"].is_number_integer()) {
                    error = "suite line " + std::to_string(line_number) +
                        " model_layer must be an integer";
                    return false;
                }
                entry.model_layer = row["model_layer"].get<int>();
                if (entry.model_layer < 1 || entry.model_layer > 92) {
                    error = "suite line " + std::to_string(line_number) +
                        " model_layer must be in 1..92";
                    return false;
                }
            }
            if (!valid_id(entry.id) || entry.split.empty() ||
                entry.text.empty() || !identifiers.insert(entry.id).second) {
                error = "suite line " + std::to_string(line_number) +
                    " has an invalid or duplicate value";
                return false;
            }
            entries.push_back(std::move(entry));
        } catch (const std::exception & exception) {
            error = "suite line " + std::to_string(line_number) + ": " +
                exception.what();
            return false;
        }
    }
    if (entries.empty()) {
        error = "suite contains no entries";
        return false;
    }
    return true;
}

bool publish_current(const std::filesystem::path & current,
                     const std::filesystem::path & destination,
                     std::string & error) {
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(current, filesystem_error) ||
        filesystem_error) {
        error = "missing current trace " + current.string();
        return false;
    }
    if (std::filesystem::exists(destination, filesystem_error) ||
        filesystem_error) {
        error = "refusing to overwrite " + destination.string();
        return false;
    }
    std::filesystem::rename(current, destination, filesystem_error);
    if (filesystem_error) {
        error = "cannot publish " + destination.string() + ": " +
            filesystem_error.message();
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <first-model-shard.gguf> <suite.jsonl> <output-dir> "
            "[gpu=0] [max-context=256] [paired=0] [core=cpu] [n-gen=1]\n",
            argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::filesystem::path suite_path = argv[2];
    const std::filesystem::path output_directory = argv[3];
    const int gpu = argc > 4 ? std::atoi(argv[4]) : 0;
    const int max_context = argc > 5 ? std::atoi(argv[5]) : 256;
    const bool paired = argc > 6 && std::atoi(argv[6]) != 0;
    const int n_gen = argc > 8 ? std::atoi(argv[8]) : 1;
    KimiK3CorePlacement core_placement = KimiK3CorePlacement::Cpu;
    if (gpu < 0 || max_context <= 0 || n_gen <= 0 || n_gen >= max_context ||
        (argc > 7 && !parse_kimi_k3_core_placement(
            argv[7], core_placement))) {
        std::fprintf(stderr, "[kimi-h16-suite] invalid argument\n");
        return 2;
    }

    std::vector<SuiteEntry> entries;
    std::string error;
    if (!read_suite(suite_path, entries, error)) {
        std::fprintf(stderr, "[kimi-h16-suite] %s\n", error.c_str());
        return 1;
    }
    if (const char * sweep = std::getenv("DFLASH_KIMI_H22_SWEEP_LAYERS");
        sweep && *sweep) {
        if (std::string(sweep) != "1" || entries.size() != 1 ||
            entries.front().model_layer != 0) {
            std::fprintf(stderr,
                "[kimi-h16-suite] H22 sweep requires one entry without model_layer\n");
            return 2;
        }
        const SuiteEntry seed = entries.front();
        entries.clear();
        for (int layer = 1; layer <= 92; ++layer) {
            SuiteEntry entry = seed;
            char suffix[8];
            std::snprintf(suffix, sizeof(suffix), "-l%02d", layer);
            entry.id += suffix;
            entry.model_layer = layer;
            entries.push_back(std::move(entry));
        }
    }
    const char * provider = std::getenv("DFLASH_KIMI_LAYER1_PROVIDER");
    const bool provider_enabled = provider && *provider &&
        std::string(provider) != "exact";
    if (paired && !provider_enabled) {
        std::fprintf(stderr,
            "[kimi-h16-suite] paired mode requires a routed provider\n");
        return 2;
    }
    const char * dynamic_layer =
        std::getenv("DFLASH_KIMI_H22_DYNAMIC_ACTIVE_LAYER");
    if (dynamic_layer && std::string(dynamic_layer) == "1" &&
        entries.front().model_layer > 0 &&
        !set_environment("DFLASH_KIMI_H22_ACTIVE_LAYER",
                         std::to_string(entries.front().model_layer))) {
        std::fprintf(stderr,
            "[kimi-h16-suite] cannot initialize dynamic model layer\n");
        return 1;
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(
        output_directory, filesystem_error);
    if (filesystem_error || std::filesystem::exists(
            output_directory / "suite-manifest.json")) {
        std::fprintf(stderr,
            "[kimi-h16-suite] output directory is unavailable or complete\n");
        return 1;
    }
    const std::filesystem::path current_teacher =
        output_directory / ".current.teacher.logits.f32";
    const std::filesystem::path current_candidate =
        output_directory / ".current.candidate.logits.f32";
    const std::filesystem::path intervention_trace =
        output_directory / "interventions.f32";
    std::filesystem::remove(current_teacher, filesystem_error);
    filesystem_error.clear();
    std::filesystem::remove(current_candidate, filesystem_error);
    if (paired &&
        (!set_environment("DFLASH_KIMI_H16_CANDIDATE_LOGITS_OUT",
                          current_candidate.string()) ||
         !set_environment("DFLASH_KIMI_LAYER1_TRACE_OUT",
                          intervention_trace.string()))) {
        std::fprintf(stderr,
            "[kimi-h16-suite] cannot configure paired trace paths\n");
        return 1;
    }

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_path.c_str())) {
        std::fprintf(stderr, "[kimi-h16-suite] tokenizer load failed\n");
        return 1;
    }
    KimiK3BackendConfig config;
    config.model_path = model_path.c_str();
    config.device.gpu = gpu;
    config.device.max_ctx = max_context;
    // Keep the owning string alive; config stores a non-owning pointer.
    const std::string teacher_trace_path = current_teacher.string();
    config.logits_trace_path = teacher_trace_path.c_str();
    config.moe_storage = MoeStoragePolicy::Ssd;
    config.expert_gpu = -1;
    config.core_placement = core_placement;
    KimiK3Backend backend(config);
    if (!backend.init()) return 1;

    json manifest;
    manifest["schema"] = "kimi-k3-h16-suite-v1";
    manifest["model_path"] = model_path;
    manifest["suite_path"] = suite_path.string();
    manifest["paired"] = paired;
    manifest["provider"] = provider_enabled ? provider : "exact";
    manifest["max_context"] = max_context;
    manifest["n_gen"] = n_gen;
    manifest["core_placement"] = argc > 7 ? argv[7] : "cpu";
    manifest["gpu"] = gpu;
    const auto record_environment = [&](const char * key) {
        if (const char * value = std::getenv(key); value && *value) {
            manifest["environment"][key] = value;
        }
    };
    record_environment("DFLASH_KIMI_LAYER1_BUDGET");
    record_environment("DFLASH_KIMI_PROVIDER_LAYER");
    record_environment("DFLASH_KIMI_LAYER1_ACTIVE_POSITION");
    record_environment("DFLASH_KIMI_H22_DYNAMIC_ACTIVE_LAYER");
    record_environment("DFLASH_KIMI_H22_SWEEP_LAYERS");
    record_environment("DFLASH_MOE_NVME_DIRECT");
    record_environment("DFLASH_MOE_NVME_DEVICE_CACHE_MB");
    record_environment("DFLASH_KIMI_CPU_THREADS");
    record_environment("DFLASH_KIMI_MMAP_DROP_PAGES");
    record_environment("KIMI_H16_REPOSITORY_COMMIT");
    record_environment("KIMI_H16_REPOSITORY_STATUS");
    record_environment("KIMI_H16_SUITE_SHA256");
    record_environment("KIMI_H17_RUNNER_SHA256");
    record_environment("KIMI_H17_QUALITY_LABEL");
    record_environment("KIMI_H17_PROVIDER_SCOPE");
    manifest["sequences"] = json::array();

    DaemonIO io;
    size_t intervention_record_start = 0;
    for (const SuiteEntry & entry : entries) {
        if (entry.model_layer > 0 &&
            !set_environment("DFLASH_KIMI_H22_ACTIVE_LAYER",
                             std::to_string(entry.model_layer))) {
            std::fprintf(stderr,
                "[kimi-h16-suite] cannot select model layer %d\n",
                entry.model_layer);
            backend.shutdown();
            return 1;
        }
        std::vector<int32_t> prompt_ids = tokenizer.encode(entry.text);
        if (prompt_ids.empty() ||
            prompt_ids.size() > static_cast<size_t>(max_context)) {
            std::fprintf(stderr,
                "[kimi-h16-suite] prompt %s has invalid token count %zu\n",
                entry.id.c_str(), prompt_ids.size());
            backend.shutdown();
            return 1;
        }
        const std::filesystem::path teacher_destination =
            output_directory / (entry.id + ".teacher.logits.f32");
        const std::filesystem::path candidate_destination =
            output_directory / (entry.id + ".candidate.logits.f32");
        if (std::filesystem::exists(teacher_destination) ||
            (paired && std::filesystem::exists(candidate_destination))) {
            std::fprintf(stderr,
                "[kimi-h16-suite] refusing existing sequence %s\n",
                entry.id.c_str());
            backend.shutdown();
            return 1;
        }
        std::filesystem::remove(current_teacher, filesystem_error);
        filesystem_error.clear();
        std::filesystem::remove(current_candidate, filesystem_error);

        GenerateRequest request;
        request.prompt = prompt_ids;
        request.n_gen = n_gen;
        request.do_sample = false;
        GenerateResult result = backend.generate(request, io);
        if (!result.ok()) {
            std::fprintf(stderr,
                "[kimi-h16-suite] sequence %s failed: %s (%s)\n",
                entry.id.c_str(),
                std::string(result.error_code()).c_str(),
                std::string(result.error_detail()).c_str());
            backend.shutdown();
            return 1;
        }
        if (!publish_current(
                current_teacher, teacher_destination, error) ||
            (paired && !publish_current(
                current_candidate, candidate_destination, error))) {
            std::fprintf(stderr, "[kimi-h16-suite] %s\n", error.c_str());
            backend.shutdown();
            return 1;
        }
        manifest["sequences"].push_back({
            {"id", entry.id},
            {"split", entry.split},
            {"text", entry.text},
            {"model_layer", entry.model_layer},
            {"prompt_tokens", prompt_ids},
            {"prompt_token_count", prompt_ids.size()},
            {"output_tokens", result.tokens},
            {"output_text", tokenizer.decode(result.tokens)},
            {"teacher_logits", teacher_destination.string()},
            {"output_logits", teacher_destination.string()},
            {"candidate_logits",
             paired ? candidate_destination.string() : ""},
            {"prefill_seconds", result.prefill_s},
            {"decode_seconds", result.decode_s},
            {"intervention_record_start",
             paired ? intervention_record_start : 0},
            {"intervention_record_count",
             paired ? prompt_ids.size() : 0},
        });
        if (paired) intervention_record_start += prompt_ids.size();
        std::fprintf(stderr,
            "[kimi-h16-suite] id=%s split=%s prompt=%zu generated=%zu "
            "prefill=%.3fs decode=%.3fs\n",
            entry.id.c_str(), entry.split.c_str(), prompt_ids.size(),
            result.tokens.size(), result.prefill_s, result.decode_s);
    }
    backend.shutdown();

    const std::filesystem::path temporary_manifest =
        output_directory / "suite-manifest.json.tmp";
    const std::filesystem::path final_manifest =
        output_directory / "suite-manifest.json";
    {
        std::ofstream output(temporary_manifest);
        output << manifest.dump(2) << '\n';
        if (!output) {
            std::fprintf(stderr,
                         "[kimi-h16-suite] cannot write manifest\n");
            return 1;
        }
    }
    std::filesystem::rename(
        temporary_manifest, final_manifest, filesystem_error);
    if (filesystem_error) {
        std::fprintf(stderr,
            "[kimi-h16-suite] cannot publish manifest: %s\n",
            filesystem_error.message().c_str());
        return 1;
    }
    std::fprintf(stderr,
        "[kimi-h16-suite] completed sequences=%zu output=%s\n",
        entries.size(), output_directory.string().c_str());
    return 0;
}
