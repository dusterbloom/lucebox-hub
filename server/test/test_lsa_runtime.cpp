#include "common/lsa_runtime.h"
#include "common/lsa_compact_retriever.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace dflash::common;

namespace {

int failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

class FixedRetriever final : public LsaRetriever {
public:
    explicit FixedRetriever(std::vector<float> scores)
        : scores_(std::move(scores)) {}

    int hidden_size() const override { return 2; }
    int key_size() const override { return 2; }

    bool score(const std::vector<float> & hidden,
               const std::vector<LsaChunk> & chunks,
               std::vector<float> & scores,
               std::string & error) override {
        ++calls;
        if ((int)hidden.size() != hidden_size()) {
            error = "hidden size mismatch";
            return false;
        }
        if (scores_.size() != chunks.size()) {
            error = "score count mismatch";
            return false;
        }
        scores = scores_;
        return true;
    }

    void set_scores(std::vector<float> scores) { scores_ = std::move(scores); }

    int calls = 0;

private:
    std::vector<float> scores_;
};

class FakeResidency final : public LsaResidency {
public:
    bool transition(const std::vector<int> & load,
                    const std::vector<int> & evict,
                    std::string & error) override {
        ++calls;
        if (fail_next) {
            fail_next = false;
            error = "injected transition failure";
            return false;
        }
        last_load = load;
        last_evict = evict;
        for (int id : load) hot.insert(id);
        for (int id : evict) hot.erase(id);
        return true;
    }

    int calls = 0;
    bool fail_next = false;
    std::set<int> hot;
    std::vector<int> last_load;
    std::vector<int> last_evict;
};

LsaChunk make_chunk(int id, float x, float y) {
    LsaChunk chunk;
    chunk.id = id;
    chunk.token_begin = id * 64;
    chunk.token_end = chunk.token_begin + 64;
    chunk.index_key = {x, y};
    return chunk;
}

void add_six_chunks(LsaRuntime & runtime) {
    std::string error;
    for (int id = 0; id < 6; ++id) {
        CHECK(runtime.add_chunk(make_chunk(id, (float)id, (float)-id), error));
    }
}

void test_topk_guards_and_interval() {
    LsaConfig config;
    config.retrieval_interval = 4;
    config.top_k = 2;
    config.attention_sink_chunks = 1;
    config.recent_chunks = 1;

    FixedRetriever retriever({0.10f, 0.90f, 0.80f, 0.70f, 0.20f, 0.30f});
    LsaRuntime runtime(config, retriever);
    add_six_chunks(runtime);

    LsaPlan plan;
    std::string error;
    CHECK(runtime.plan(0, {1.0f, 0.0f}, plan, error));
    CHECK(plan.triggered);
    CHECK(plan.keep == std::vector<int>({0, 1, 2, 5}));
    CHECK(plan.load == plan.keep);
    CHECK(plan.evict.empty());

    FakeResidency residency;
    CHECK(runtime.commit(plan, residency, error));
    CHECK(runtime.hot_chunks() == plan.keep);
    CHECK(residency.hot == std::set<int>({0, 1, 2, 5}));

    LsaPlan skipped;
    CHECK(runtime.plan(1, {1.0f, 0.0f}, skipped, error));
    CHECK(!skipped.triggered);
    CHECK(retriever.calls == 1);

    retriever.set_scores({0.10f, 0.20f, 0.30f, 0.95f, 0.85f, 0.40f});
    LsaPlan second;
    CHECK(runtime.plan(4, {0.0f, 1.0f}, second, error));
    CHECK(second.keep == std::vector<int>({0, 3, 4, 5}));
    CHECK(second.load == std::vector<int>({3, 4}));
    CHECK(second.evict == std::vector<int>({1, 2}));
    CHECK(runtime.commit(second, residency, error));
    CHECK(residency.hot == std::set<int>({0, 3, 4, 5}));

    const auto stats = runtime.stats();
    CHECK(stats.retrieval_cycles == 2);
    CHECK(stats.loaded_chunks == 6);
    CHECK(stats.evicted_chunks == 2);
    CHECK(stats.max_hot_chunks == 4);
}

void test_failed_transition_is_transactional() {
    LsaConfig config;
    config.retrieval_interval = 1;
    config.top_k = 1;
    config.attention_sink_chunks = 0;
    config.recent_chunks = 0;

    FixedRetriever retriever({0.90f, 0.10f});
    LsaRuntime runtime(config, retriever);
    std::string error;
    CHECK(runtime.add_chunk(make_chunk(0, 1.0f, 0.0f), error));
    CHECK(runtime.add_chunk(make_chunk(1, 0.0f, 1.0f), error));

    FakeResidency residency;
    residency.fail_next = true;
    LsaPlan plan;
    CHECK(runtime.plan(0, {1.0f, 0.0f}, plan, error));
    CHECK(!runtime.commit(plan, residency, error));
    CHECK(runtime.hot_chunks().empty());
    CHECK(runtime.stats().retrieval_cycles == 0);

    CHECK(runtime.commit(plan, residency, error));
    CHECK(runtime.hot_chunks() == std::vector<int>({0}));
}

void test_threshold_selection_and_validation() {
    LsaConfig config;
    config.retrieval_interval = 8;
    config.top_k = 0;
    config.threshold = 0.75f;
    config.attention_sink_chunks = 0;
    config.recent_chunks = 0;

    FixedRetriever retriever({0.74f, 0.75f, 0.76f});
    LsaRuntime runtime(config, retriever);
    std::string error;
    CHECK(runtime.add_chunk(make_chunk(0, 1.0f, 0.0f), error));
    CHECK(runtime.add_chunk(make_chunk(1, 0.0f, 1.0f), error));
    CHECK(runtime.add_chunk(make_chunk(2, 1.0f, 1.0f), error));

    LsaPlan plan;
    CHECK(runtime.plan(0, {1.0f, 1.0f}, plan, error));
    CHECK(plan.keep == std::vector<int>({1, 2}));

    LsaChunk bad = make_chunk(3, 1.0f, 1.0f);
    bad.index_key.push_back(2.0f);
    CHECK(!runtime.add_chunk(std::move(bad), error));
    CHECK(error.find("index key") != std::string::npos);

    LsaPlan bad_hidden;
    CHECK(!runtime.plan(8, {1.0f}, bad_hidden, error));
    CHECK(error.find("hidden") != std::string::npos);
}

void test_interval_crossing_and_chunk_order() {
    LsaConfig config;
    config.retrieval_interval = 64;
    config.top_k = 1;
    config.attention_sink_chunks = 0;
    config.recent_chunks = 0;

    FixedRetriever retriever({0.90f, 0.10f});
    LsaRuntime runtime(config, retriever);
    std::string error;
    CHECK(runtime.add_chunk(make_chunk(0, 1.0f, 0.0f), error));
    CHECK(runtime.add_chunk(make_chunk(1, 0.0f, 1.0f), error));

    FakeResidency residency;
    LsaPlan initial;
    CHECK(runtime.plan(0, {1.0f, 0.0f}, initial, error));
    CHECK(initial.triggered);
    CHECK(runtime.commit(initial, residency, error));

    LsaPlan before_boundary;
    CHECK(runtime.plan(63, {1.0f, 0.0f}, before_boundary, error));
    CHECK(!before_boundary.triggered);

    // Speculative decode may commit across 64 without ever presenting exactly 64.
    LsaPlan crossed_boundary;
    CHECK(runtime.plan(70, {1.0f, 0.0f}, crossed_boundary, error));
    CHECK(crossed_boundary.triggered);
    CHECK(runtime.commit(crossed_boundary, residency, error));

    LsaPlan repeated;
    CHECK(runtime.plan(70, {1.0f, 0.0f}, repeated, error));
    CHECK(!repeated.triggered);

    LsaChunk overlapping = make_chunk(2, 0.0f, 0.0f);
    overlapping.token_begin = 96;
    overlapping.token_end = 160;
    CHECK(!runtime.add_chunk(std::move(overlapping), error));
    CHECK(error.find("token order") != std::string::npos);
}

void test_compact_retriever_scores_matching_key_first() {
    LsaCompactConfig config;
    config.hidden_size = 2;
    config.rank = 2;
    config.kv_heads = 1;
    config.head_dim = 2;
    config.decision_threshold = 0.0f;
    LsaCompactRetriever retriever(config);
    std::string error;
    CHECK(retriever.set_weights(
        {1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 1.0f},
        error));

    std::vector<LsaChunk> chunks = {
        make_chunk(0, 1.0f, 0.0f),
        make_chunk(1, 0.0f, 1.0f),
    };
    std::vector<float> scores;
    CHECK(retriever.score({1.0f, 0.0f}, chunks, scores, error));
    CHECK(scores.size() == 2);
    CHECK(scores[0] > scores[1]);

    LsaCompactRetriever unloaded(config);
    CHECK(!unloaded.score({1.0f, 0.0f}, chunks, scores, error));
    CHECK(error.find("not loaded") != std::string::npos);
}

uint64_t fnv1a64(const std::vector<uint16_t> & values) {
    uint64_t checksum = 14695981039346656037ULL;
    const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
    for (size_t index = 0;
         index < values.size() * sizeof(uint16_t); ++index) {
        checksum ^= bytes[index];
        checksum *= 1099511628211ULL;
    }
    return checksum;
}

void test_compact_retriever_artifact_contract() {
    namespace fs = std::filesystem;
    const fs::path directory =
        fs::temp_directory_path() / "lsa-compact-artifact-test";
    fs::remove_all(directory);
    fs::create_directories(directory);
    const std::vector<uint16_t> weights = {
        0x3c00, 0x0000, 0x0000, 0x3c00,
        0x3c00, 0x0000, 0x0000, 0x3c00,
    };
    {
        std::ofstream output(
            directory / "encoder.f16.bin", std::ios::binary);
        output.write(reinterpret_cast<const char *>(weights.data()),
                     static_cast<std::streamsize>(
                         weights.size() * sizeof(uint16_t)));
    }
    std::ostringstream checksum;
    checksum << std::hex << std::setfill('0') << std::setw(16)
             << fnv1a64(weights);
    nlohmann::json manifest = {
        {"schema", "luce.lsa.qwen35.encoder.v1"},
        {"dataset",
         {{"hidden_size", 2}, {"kv_heads", 1}, {"head_dim", 2}}},
        {"rank", 2},
        {"score_temperature", 0.1},
        {"decision_threshold", 0.0},
        {"logit_scale", 12.0},
        {"weight_file",
         {{"name", "encoder.f16.bin"},
          {"dtype", "float16-le"},
          {"fnv1a64", checksum.str()},
          {"size_bytes", weights.size() * sizeof(uint16_t)},
          {"layout",
           {{{"name", "down.weight"},
             {"shape", {2, 2}},
             {"offset_bytes", 0}},
            {{"name", "up.weight"},
             {"shape", {2, 2}},
             {"offset_bytes", 4 * sizeof(uint16_t)}}}}}},
    };
    {
        std::ofstream output(directory / "encoder.json");
        output << manifest.dump(2) << '\n';
    }

    LsaCompactRetriever retriever;
    std::string error;
    CHECK(retriever.load_artifact(directory.string(), error));
    CHECK(retriever.hidden_size() == 2);
    CHECK(retriever.key_size() == 2);
    std::vector<float> scores;
    CHECK(retriever.score(
        {1.0f, 0.0f}, {make_chunk(0, 1.0f, 0.0f)}, scores, error));

    {
        std::fstream output(
            directory / "encoder.f16.bin",
            std::ios::binary | std::ios::in | std::ios::out);
        const uint16_t corrupt = 0;
        output.write(reinterpret_cast<const char *>(&corrupt),
                     sizeof(corrupt));
    }
    LsaCompactRetriever corrupt;
    CHECK(!corrupt.load_artifact(directory.string(), error));
    CHECK(error.find("checksum") != std::string::npos);
    fs::remove_all(directory);
}

}  // namespace

int main() {
    test_topk_guards_and_interval();
    test_failed_transition_is_transactional();
    test_threshold_selection_and_validation();
    test_interval_crossing_and_chunk_order();
    test_compact_retriever_scores_matching_key_first();
    test_compact_retriever_artifact_contract();
    if (failures != 0) {
        std::fprintf(stderr, "%d LSA runtime test(s) failed\n", failures);
        return 1;
    }
    std::printf("LSA runtime tests passed\n");
    return 0;
}
