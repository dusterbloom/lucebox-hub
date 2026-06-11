#pragma once

#include "qwen35_lsa_capture.h"

#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct Qwen35LsaRawWriterConfig {
    std::string output_dir;
    std::string model_fingerprint = "unknown";
    int hidden_size = 5120;
    int kv_heads = 4;
    int query_heads = 24;
    int head_dim = 256;
    int block_size = 64;
    int lookahead_horizon = 64;
    int key_layer = 47;
    std::vector<int> oracle_layers;
};

class Qwen35LsaRawWriter {
public:
    Qwen35LsaRawWriter();
    ~Qwen35LsaRawWriter();
    Qwen35LsaRawWriter(Qwen35LsaRawWriter &&) noexcept;
    Qwen35LsaRawWriter & operator=(Qwen35LsaRawWriter &&) noexcept;

    Qwen35LsaRawWriter(const Qwen35LsaRawWriter &) = delete;
    Qwen35LsaRawWriter & operator=(const Qwen35LsaRawWriter &) = delete;

    bool open(const Qwen35LsaRawWriterConfig & config, std::string & error);

    // Chunks must be contiguous. Historical post-RoPE K and the key-layer
    // pre-RoPE K are always appended. When store_boundary_example is true,
    // the current chunk supplies future Q for boundary=chunk_start and the
    // preceding chunk's final hidden state supplies the query embedding.
    bool append(int chunk_start,
                const Qwen35LsaCaptureBatch & batch,
                bool store_boundary_example,
                std::string & error);

    bool finalize(std::string & error);
    bool is_open() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dflash::common
