// Lookahead Sparse Attention runtime policy.
//
// The retriever and KV residency implementation are injected so the selection
// policy can be tested independently of model weights and GPU backends.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct LsaConfig {
    int   retrieval_interval = 64;
    int   top_k = 0;
    float threshold = 0.5f;
    int   attention_sink_chunks = 1;
    int   recent_chunks = 128;
};

struct LsaChunk {
    int id = -1;
    int token_begin = 0;
    int token_end = 0;
    std::vector<float> index_key;
};

struct LsaPlan {
    bool triggered = false;
    std::vector<int> keep;
    std::vector<int> load;
    std::vector<int> evict;

private:
    friend class LsaRuntime;
    uint64_t catalog_version = 0;
    uint64_t hot_version = 0;
    int committed_tokens = 0;
};

struct LsaStats {
    uint64_t retrieval_cycles = 0;
    uint64_t loaded_chunks = 0;
    uint64_t evicted_chunks = 0;
    uint64_t max_hot_chunks = 0;
};

class LsaRetriever {
public:
    virtual ~LsaRetriever() = default;

    virtual int hidden_size() const = 0;
    virtual int key_size() const = 0;
    virtual bool score(const std::vector<float> & hidden,
                       const std::vector<LsaChunk> & chunks,
                       std::vector<float> & scores,
                       std::string & error) = 0;
};

class LsaResidency {
public:
    virtual ~LsaResidency() = default;

    virtual bool transition(const std::vector<int> & load,
                            const std::vector<int> & evict,
                            std::string & error) = 0;
};

class LsaRuntime {
public:
    LsaRuntime(LsaConfig config, LsaRetriever & retriever);

    bool add_chunk(LsaChunk chunk, std::string & error);
    void clear();

    bool plan(int committed_tokens, const std::vector<float> & hidden,
              LsaPlan & out, std::string & error);
    bool commit(const LsaPlan & plan, LsaResidency & residency,
                std::string & error);

    const std::vector<int> & hot_chunks() const { return hot_chunks_; }
    const LsaStats & stats() const { return stats_; }
    const std::vector<LsaChunk> & chunks() const { return chunks_; }

private:
    LsaConfig config_;
    LsaRetriever & retriever_;
    std::vector<LsaChunk> chunks_;
    std::vector<int> hot_chunks_;
    LsaStats stats_;
    uint64_t catalog_version_ = 0;
    uint64_t hot_version_ = 0;
    int next_trigger_token_ = 0;
};

}  // namespace dflash::common
