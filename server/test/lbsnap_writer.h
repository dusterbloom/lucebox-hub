// Shared LBSNAP01 writer + prompt loading for the bridge paired-state tools
// (luce_box lucebox_rnd/KV_translator_qwen_ds4). Read by extract/lbsnap.py.
//
// The format is deliberately tiny and self-describing so both the DS4 label
// dump and the Qwen source dump share one writer and one Python reader.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "server/tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace lbsnap {

inline bool write_bytes(FILE * f, const void * data, size_t n) {
    return n == 0 || std::fwrite(data, 1, n, f) == n;
}

inline bool write_u32(FILE * f, uint32_t v) { return write_bytes(f, &v, sizeof(v)); }
inline bool write_u64(FILE * f, uint64_t v) { return write_bytes(f, &v, sizeof(v)); }

inline bool write_str(FILE * f, const std::string & s) {
    return write_u32(f, (uint32_t) s.size()) &&
           write_bytes(f, s.data(), s.size());
}

inline bool write_tensor(FILE * f, ggml_tensor * t) {
    const std::string name = ggml_get_name(t) ? ggml_get_name(t) : "";
    if (!write_str(f, name)) return false;
    if (!write_str(f, ggml_type_name(t->type))) return false;
    if (!write_u32(f, (uint32_t) GGML_MAX_DIMS)) return false;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (!write_u32(f, (uint32_t) t->ne[d])) return false;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (!write_u64(f, (uint64_t) nbytes)) return false;
    std::vector<uint8_t> bytes(nbytes);
    ggml_backend_tensor_get(t, bytes.data(), 0, nbytes);
    return write_bytes(f, bytes.data(), nbytes);
}

// Write every tensor in `ctx` with the snapshot position and prompt length.
// Returns the tensor count, or -1 on write failure.
inline long long write_container(const std::string & path, ggml_context * ctx,
                                 int cur_pos, int prompt_len) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", path.c_str());
        return -1;
    }
    const char magic[8] = {'L', 'B', 'S', 'N', 'A', 'P', '0', '1'};
    bool ok = write_bytes(f, magic, sizeof(magic)) &&
              write_u32(f, (uint32_t) cur_pos) &&
              write_u32(f, (uint32_t) prompt_len);

    uint32_t n_tensors = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; 
         t = ggml_get_next_tensor(ctx, t)) {
        ++n_tensors;
    }
    ok = ok && write_u32(f, n_tensors);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t && ok;
         t = ggml_get_next_tensor(ctx, t)) {
        ok = write_tensor(f, t);
    }
    std::fclose(f);
    if (!ok) {
        std::fprintf(stderr, "FAIL: write error for %s\n", path.c_str());
        return -1;
    }
    std::fprintf(stderr, "[dump] %s: %u tensors, cur_pos=%d, prompt=%d\n",
                 path.c_str(), n_tensors, cur_pos, prompt_len);
    return n_tensors;
}

inline std::vector<int32_t> read_token_file(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return {};
    std::vector<int32_t> out;
    int32_t v;
    while (std::fread(&v, sizeof(v), 1, f) == 1) out.push_back(v);
    std::fclose(f);
    return out;
}

// A *.txt input is raw prompt text: tokenize it with the model's own vocab so
// the corpus builder needs no second tokenizer implementation. Anything else
// is a raw little-endian int32 token stream.
inline std::vector<int32_t> read_prompt(const char * path,
                                        const std::string & model) {
    const std::string p = path;
    if (p.size() < 4 || p.compare(p.size() - 4, 4, ".txt") != 0) {
        return read_token_file(path);
    }
    FILE * f = std::fopen(path, "rb");
    if (!f) return {};
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    std::fclose(f);

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model.c_str())) {
        std::fprintf(stderr, "FAIL: tokenizer load from %s\n", model.c_str());
        return {};
    }
    return tokenizer.encode(text);
}

}  // namespace lbsnap
