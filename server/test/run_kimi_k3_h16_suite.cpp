#include "kimi_k3/kimi_k3_backend.h"
#include "server/chat_template.h"
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

bool write_manifest_checkpoint(const json & manifest,
                               const std::filesystem::path & output_directory,
                               std::string & error) {
    const std::filesystem::path temporary =
        output_directory / "suite-manifest.partial.json.tmp";
    const std::filesystem::path checkpoint =
        output_directory / "suite-manifest.partial.json";
    {
        std::ofstream output(temporary);
        output << manifest.dump(2) << '\n';
        if (!output) {
            error = "cannot write partial suite manifest";
            return false;
        }
    }
    std::error_code filesystem_error;
    std::filesystem::rename(temporary, checkpoint, filesystem_error);
    if (filesystem_error) {
        error = "cannot publish partial suite manifest: " +
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
            "[gpu=0] [max-context=256] [paired=0] [core=cpu] [n-gen=1] "
            "[draft.gguf] [draft-gpu=0]\n",
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
    const std::string draft_path = argc > 9 ? argv[9] : "";
    const int draft_gpu = argc > 10 ? std::atoi(argv[10]) : gpu;
    const char * disable_logits_environment =
        std::getenv("DFLASH_KIMI_SUITE_DISABLE_LOGITS");
    const bool record_logits = !disable_logits_environment ||
        std::string(disable_logits_environment) != "1";
    const char * resume_environment =
        std::getenv("DFLASH_KIMI_H16_RESUME");
    const bool resume = resume_environment &&
        std::string(resume_environment) == "1";
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
    const char * chat_template_environment =
        std::getenv("DFLASH_KIMI_H16_CHAT_TEMPLATE");
    const bool use_chat_template = chat_template_environment &&
        std::string(chat_template_environment) == "1";
    const char * thinking_environment =
        std::getenv("DFLASH_KIMI_H16_ENABLE_THINKING");
    const bool enable_thinking = thinking_environment &&
        std::string(thinking_environment) == "1";
    if (enable_thinking && !use_chat_template) {
        std::fprintf(stderr,
            "[kimi-h16-suite] thinking requires the GGUF chat template\n");
        return 2;
    }
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
    const std::filesystem::path partial_manifest =
        output_directory / "suite-manifest.partial.json";
    if (std::filesystem::exists(partial_manifest) && !resume) {
        std::fprintf(stderr,
            "[kimi-h16-suite] partial manifest exists; set "
            "DFLASH_KIMI_H16_RESUME=1 to resume\n");
        return 1;
    }
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
    if (use_chat_template && tokenizer.chat_template().empty()) {
        std::fprintf(stderr,
            "[kimi-h16-suite] GGUF chat template requested but unavailable\n");
        return 1;
    }
    KimiK3BackendConfig config;
    config.model_path = model_path.c_str();
    config.draft_path = draft_path.empty() ? nullptr : draft_path.c_str();
    config.draft_gpu = draft_gpu;
    config.device.gpu = gpu;
    config.device.max_ctx = max_context;
    // Keep the owning string alive; config stores a non-owning pointer.
    const std::string teacher_trace_path = record_logits
        ? current_teacher.string() : std::string();
    config.logits_trace_path = record_logits
        ? teacher_trace_path.c_str() : nullptr;
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
    manifest["chat_template"] =
        use_chat_template
            ? (enable_thinking ? "gguf-jinja-thinking-on"
                               : "gguf-jinja-thinking-off")
            : "raw-text";
    manifest["thinking_enabled"] = enable_thinking;
    manifest["max_context"] = max_context;
    manifest["n_gen"] = n_gen;
    manifest["draft_path"] = draft_path;
    manifest["draft_gpu"] = draft_gpu;
    manifest["logits_recorded"] = record_logits;
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
    record_environment("DFLASH_KIMI_H22_LAYER_BUDGETS");
    record_environment("DFLASH_KIMI_H16_CHAT_TEMPLATE");
    record_environment("DFLASH_KIMI_H16_ENABLE_THINKING");
    record_environment("DFLASH_MOE_NVME_DIRECT");
    record_environment("DFLASH_MOE_NVME_DEVICE_CACHE_MB");
    record_environment("DFLASH_KIMI_CPU_THREADS");
    record_environment("DFLASH_KIMI_MMAP_DROP_PAGES");
    record_environment("DFLASH_KIMI_MOE_CORE_OFFLOAD");
    record_environment("DFLASH_KIMI_DRAFT_MAX_BLOCK");
    record_environment("DFLASH_KIMI_DRAFT_DELAY_TOKENS");
    record_environment("DFLASH_KIMI_S0_SERIAL_CORE_ROWS");
    record_environment("DFLASH_KIMI_S0_SERIAL_EXPERT_ROWS");
    record_environment("DFLASH_KIMI_P56_PREFILL_CENSUS");
    record_environment("DFLASH_KIMI_SUITE_DISABLE_LOGITS");
    record_environment("DFLASH_KIMI_H16_RESUME");
    record_environment("KIMI_H16_REPOSITORY_COMMIT");
    record_environment("KIMI_H16_REPOSITORY_STATUS");
    record_environment("KIMI_H16_SUITE_SHA256");
    record_environment("KIMI_H17_RUNNER_SHA256");
    record_environment("KIMI_H17_QUALITY_LABEL");
    record_environment("KIMI_H17_PROVIDER_SCOPE");
    manifest["sequences"] = json::array();

    size_t resume_count = 0;
    if (resume && std::filesystem::exists(partial_manifest)) {
        json resumed;
        try {
            std::ifstream input(partial_manifest);
            input >> resumed;
        } catch (const std::exception & exception) {
            std::fprintf(stderr,
                "[kimi-h16-suite] cannot parse partial manifest: %s\n",
                exception.what());
            backend.shutdown();
            return 1;
        }
        static const std::vector<const char *> identity_keys = {
            "schema", "model_path", "suite_path", "paired", "provider",
            "chat_template", "max_context", "n_gen", "draft_path",
            "draft_gpu", "logits_recorded", "core_placement", "gpu",
        };
        for (const char * key : identity_keys) {
            if (!resumed.contains(key) || resumed[key] != manifest[key]) {
                std::fprintf(stderr,
                    "[kimi-h16-suite] resume identity mismatch: %s\n", key);
                backend.shutdown();
                return 1;
            }
        }
        if (!resumed.contains("sequences") ||
            !resumed["sequences"].is_array() ||
            resumed["sequences"].size() > entries.size()) {
            std::fprintf(stderr,
                "[kimi-h16-suite] invalid partial sequence list\n");
            backend.shutdown();
            return 1;
        }
        resume_count = resumed["sequences"].size();
        for (size_t index = 0; index < resume_count; ++index) {
            const json & row = resumed["sequences"][index];
            if (!row.contains("id") || !row.contains("split") ||
                !row.contains("text") || !row.contains("model_layer") ||
                row["id"] != entries[index].id ||
                row["split"] != entries[index].split ||
                row["text"] != entries[index].text ||
                row["model_layer"] != entries[index].model_layer) {
                std::fprintf(stderr,
                    "[kimi-h16-suite] resume sequence mismatch at %zu\n",
                    index);
                backend.shutdown();
                return 1;
            }
        }
        manifest = std::move(resumed);
        std::fprintf(stderr,
            "[kimi-h16-suite] resuming completed sequences=%zu\n",
            resume_count);
    }

    DaemonIO io;
    size_t intervention_record_start = 0;
    if (paired) {
        for (const json & row : manifest["sequences"]) {
            intervention_record_start +=
                row.value("intervention_record_count", size_t{0});
        }
    }
    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index) {
        const SuiteEntry & entry = entries[entry_index];
        if (entry.model_layer > 0 &&
            !set_environment("DFLASH_KIMI_H22_ACTIVE_LAYER",
                             std::to_string(entry.model_layer))) {
            std::fprintf(stderr,
                "[kimi-h16-suite] cannot select model layer %d\n",
                entry.model_layer);
            backend.shutdown();
            return 1;
        }
        std::string rendered_prompt = entry.text;
        if (use_chat_template) {
            const std::string bos = tokenizer.bos_id() >= 0
                ? tokenizer.raw_token(tokenizer.bos_id()) : std::string();
            const std::string eos = tokenizer.eos_id() >= 0
                ? tokenizer.raw_token(tokenizer.eos_id()) : std::string();
            try {
                rendered_prompt = render_chat_template_jinja(
                    tokenizer.chat_template(),
                    {{"user", entry.text, ""}}, bos, eos,
                    /*add_generation_prompt=*/true,
                    enable_thinking, "");
            } catch (const std::exception & exception) {
                std::fprintf(stderr,
                    "[kimi-h16-suite] chat template failed for %s: %s\n",
                    entry.id.c_str(), exception.what());
                backend.shutdown();
                return 1;
            }
        }
        std::vector<int32_t> prompt_ids = tokenizer.encode(rendered_prompt);
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
        if (entry_index < resume_count) {
            const json & row = manifest["sequences"][entry_index];
            const bool teacher_ok = !record_logits ||
                std::filesystem::is_regular_file(teacher_destination);
            const bool candidate_ok = !paired ||
                std::filesystem::is_regular_file(candidate_destination);
            if (!teacher_ok || !candidate_ok ||
                !row.contains("prompt_tokens") ||
                row["prompt_tokens"] != prompt_ids) {
                std::fprintf(stderr,
                    "[kimi-h16-suite] resumed artifact mismatch for %s\n",
                    entry.id.c_str());
                backend.shutdown();
                return 1;
            }
            std::fprintf(stderr,
                "[kimi-h16-suite] resume skip id=%s\n", entry.id.c_str());
            continue;
        }
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
        if ((record_logits && !publish_current(
                current_teacher, teacher_destination, error)) ||
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
            {"rendered_prompt", rendered_prompt},
            {"model_layer", entry.model_layer},
            {"prompt_tokens", prompt_ids},
            {"prompt_token_count", prompt_ids.size()},
            {"output_tokens", result.tokens},
            {"output_text", tokenizer.decode(result.tokens)},
            {"teacher_logits",
             record_logits ? teacher_destination.string() : ""},
            {"output_logits",
             record_logits ? teacher_destination.string() : ""},
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
        if (!write_manifest_checkpoint(manifest, output_directory, error)) {
            std::fprintf(stderr, "[kimi-h16-suite] %s\n", error.c_str());
            backend.shutdown();
            return 1;
        }
        std::fprintf(stderr,
            "[kimi-h16-suite] id=%s split=%s prompt=%zu generated=%zu "
            "prefill=%.3fs decode=%.3fs\n",
            entry.id.c_str(), entry.split.c_str(), prompt_ids.size(),
            result.tokens.size(), result.prefill_s, result.decode_s);
    }
    backend.shutdown();

    const std::filesystem::path final_manifest =
        output_directory / "suite-manifest.json";
    if (!std::filesystem::exists(partial_manifest) &&
        !write_manifest_checkpoint(manifest, output_directory, error)) {
        std::fprintf(stderr, "[kimi-h16-suite] %s\n", error.c_str());
        return 1;
    }
    std::filesystem::rename(
        partial_manifest, final_manifest, filesystem_error);
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
