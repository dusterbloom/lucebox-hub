// Paired-state extraction: dump a native DS4 snapshot to a portable container
// for the Qwen->DS4 bridge spike (luce_box lucebox_rnd/KV_translator_qwen_ds4).
//
// Prefills a prompt (token IDs supplied by the Python corpus builder, so this
// tool is tokenizer-free), saves a checkpoint at `cut`, and writes every
// snapshot tensor in the container read by extract/lbsnap.py.
//
// Model-gated, not registered in ordinary CI.
//
//   ds4_snapshot_dump MODEL.gguf TOKENS.bin CUT OUT.lbsnap [GPU]
//
//   TOKENS.bin   little-endian int32 token IDs (the target tokenizer's)
//   CUT          absolute position to checkpoint; <=0 or >n means end-of-prompt
//   OUT.lbsnap   LBSNAP01 container (see extract/lbsnap.py)

#include "deepseek4/deepseek4_backend.h"
#include "server/tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

bool write_bytes(FILE * f, const void * data, size_t n) {
    return n == 0 || std::fwrite(data, 1, n, f) == n;
}

bool write_u32(FILE * f, uint32_t v) { return write_bytes(f, &v, sizeof(v)); }
bool write_u64(FILE * f, uint64_t v) { return write_bytes(f, &v, sizeof(v)); }

bool write_str(FILE * f, const std::string & s) {
    return write_u32(f, (uint32_t) s.size()) &&
           write_bytes(f, s.data(), s.size());
}

size_t tensor_nelements(const ggml_tensor * t) {
    size_t n = 1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) n *= (size_t) t->ne[d];
    return n;
}

bool write_tensor(FILE * f, ggml_tensor * t) {
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

std::vector<int32_t> read_tokens(const char * path) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return {};
    std::vector<int32_t> out;
    int32_t v;
    while (std::fread(&v, sizeof(v), 1, f) == 1) out.push_back(v);
    std::fclose(f);
    return out;
}

// A *.txt input is raw prompt text: tokenize it with the model's own vocab so
// the corpus builder does not need a second tokenizer implementation.
// Anything else is a raw little-endian int32 token stream.
std::vector<int32_t> read_prompt(const char * path, const std::string & model) {
    const std::string p = path;
    if (p.size() >= 4 && p.compare(p.size() - 4, 4, ".txt") == 0) {
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
    return read_tokens(path);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s MODEL.gguf TOKENS.bin CUT OUT.lbsnap [GPU]\n",
                     argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const std::string tokens_path = argv[2];
    const int cut_arg = std::atoi(argv[3]);
    const std::string out_path = argv[4];
    const int gpu = argc > 5 ? std::atoi(argv[5]) : 0;

    const std::vector<int32_t> prompt = read_prompt(tokens_path.c_str(), model);
    if (prompt.empty()) {
        std::fprintf(stderr, "FAIL: no tokens read from %s\n",
                     tokens_path.c_str());
        return 1;
    }

    DeepSeek4BackendConfig config;
    config.model_path = model;
#if defined(DFLASH27B_BACKEND_HIP)
    config.device.backend = PlacementBackend::Hip;
#else
    config.device.backend = PlacementBackend::Cuda;
#endif
    config.device.gpu = gpu;
    config.paged_attention = false;
    config.max_concurrency = 1;
    config.prefill_mode = PrefillAttentionMode::Exact;
    config.chunk = 512;
    config.max_ctx = (int) prompt.size() + 256;

    DeepSeek4Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }

    int cut = cut_arg;
    if (cut <= 0 || cut > (int) prompt.size()) cut = (int) prompt.size();

    GenerateRequest req;
    req.prompt = prompt;
    req.n_gen = 0;
    req.snap_slot = 0;
    req.snap_pos = cut;
    DaemonIO io;
    if (!backend.generate(req, io).ok() || !backend.snapshot_used(0) ||
        backend.snapshot_cur_pos(0) != cut) {
        std::fprintf(stderr, "FAIL: checkpoint at cut=%d failed\n", cut);
        return 1;
    }

    const ModelBackend::SnapshotRef ref = backend.snapshot_ref(0);
    if (!ref.ctx || !ref.buf) {
        std::fprintf(stderr, "FAIL: snapshot_ref(0) empty\n");
        return 1;
    }

    FILE * f = std::fopen(out_path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot open %s\n", out_path.c_str());
        return 1;
    }
    const char magic[8] = {'L', 'B', 'S', 'N', 'A', 'P', '0', '1'};
    bool ok = write_bytes(f, magic, sizeof(magic)) &&
              write_u32(f, (uint32_t) ref.cur_pos) &&
              write_u32(f, (uint32_t) prompt.size());

    uint32_t n_tensors = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t && ok;
         t = ggml_get_next_tensor(ref.ctx, t)) {
        ++n_tensors;
    }
    ok = ok && write_u32(f, n_tensors);

    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t && ok;
         t = ggml_get_next_tensor(ref.ctx, t)) {
        ok = write_tensor(f, t);
    }
    std::fclose(f);

    if (!ok) {
        std::fprintf(stderr, "FAIL: write error\n");
        return 1;
    }
    std::fprintf(stderr,
                 "[dump] %s: %u tensors, cur_pos=%d, prompt=%zu\n",
                 out_path.c_str(), n_tensors, ref.cur_pos, prompt.size());
    return 0;
}
