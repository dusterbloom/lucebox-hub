// Requantize selected dense tensors of a GGUF shard from Q8_0 to a target quant type
// (Q6_K / Q5_K), copying everything else byte-for-byte. Metadata and split layout
// are preserved exactly.
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

static bool should_requant(const std::string &name) {
    if (name.find("attn_qkv") != std::string::npos) return true;
    if (name.find("attn_gate") != std::string::npos) return true;
    if (name.find("ssm_out") != std::string::npos) return true;
    if (name == "output.weight") return true;
    if (name.find("attn_output") != std::string::npos) return true;
    if (name.find("ffn_gate_shexp") != std::string::npos) return true;
    if (name.find("ffn_up_shexp") != std::string::npos) return true;
    if (name.find("ffn_down_shexp") != std::string::npos) return true;
    if (name.find("attn_k.weight") != std::string::npos) return true;
    if (name.find("attn_v.weight") != std::string::npos) return true;
    if (name.find("attn_q.weight") != std::string::npos) return true;
    return false;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <in.gguf> <out.gguf> <Q6_K|Q5_K>\n", argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    std::string type_name = argv[3];
    ggml_type new_type;
    if (type_name == "Q6_K") new_type = GGML_TYPE_Q6_K;
    else if (type_name == "Q5_K") new_type = GGML_TYPE_Q5_K;
    else { fprintf(stderr, "unsupported type %s\n", type_name.c_str()); return 1; }

    ggml_context *meta_ctx = nullptr;
    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&meta_ctx };
    gguf_context *src = gguf_init_from_file(in_path, params);
    if (!src) { fprintf(stderr, "failed to open %s\n", in_path); return 1; }

    int64_t n_tensors = gguf_get_n_tensors(src);
    size_t src_data_offset = gguf_get_data_offset(src);

    gguf_context *out = gguf_init_empty();
    gguf_set_kv(out, src);

    // small throwaway ctx to build ggml_tensor metadata for gguf_add_tensor (no_alloc)
    ggml_init_params gip = { /*mem_size=*/ (size_t)(64*1024*1024), /*mem_buffer=*/nullptr, /*no_alloc=*/true };
    ggml_context *shape_ctx = ggml_init(gip);

    std::vector<ggml_type> new_types(n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(src, i);
        ggml_type old_type = gguf_get_tensor_type(src, i);
        ggml_type t = old_type;
        ggml_tensor *src_t0 = ggml_get_tensor(meta_ctx, name);
        if (!src_t0) { fprintf(stderr, "missing tensor meta for %s\n", name); return 1; }
        if (old_type == GGML_TYPE_Q8_0 && should_requant(name)) {
            // K-quants need the row (ne[0]) divisible by the 256-element superblock.
            // Tensors that don't fit (e.g. small attn_k/attn_v rows) stay at Q8_0.
            if (src_t0->ne[0] % 256 == 0) {
                t = new_type;
            } else {
                fprintf(stderr, "skip %-40s ne0=%lld not divisible by 256, keeping Q8_0\n", name, (long long)src_t0->ne[0]);
            }
        }
        new_types[i] = t;
        ggml_tensor *src_t = src_t0;
        ggml_tensor *nt = ggml_new_tensor(shape_ctx, t, GGML_MAX_DIMS, src_t->ne);
        ggml_set_name(nt, name);
        gguf_add_tensor(out, nt);
    }

    if (!gguf_write_to_file(out, out_path, /*only_meta=*/true)) {
        fprintf(stderr, "failed writing meta to %s\n", out_path);
        return 1;
    }

    FILE *fin = fopen(in_path, "rb");
    if (!fin) { fprintf(stderr, "cannot reopen %s\n", in_path); return 1; }
    FILE *fout = fopen(out_path, "r+b");
    if (!fout) { fprintf(stderr, "cannot reopen %s\n", out_path); return 1; }

    // gguf_get_data_offset() on a freshly-built (gguf_init_empty) context returns the
    // stale default (0), since gguf_write_out() never updates ctx->offset. The meta-only
    // write we just performed leaves the file exactly at the real data offset, so use that.
    if (fseeko(fout, 0, SEEK_END) != 0) { perror("fseeko out end"); return 1; }
    size_t out_data_offset = (size_t)ftello(fout);

    size_t alignment = gguf_get_alignment(out);
    int n_changed = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(src, i);
        ggml_type old_type = gguf_get_tensor_type(src, i);
        size_t old_size = gguf_get_tensor_size(src, i);
        size_t old_offset = gguf_get_tensor_offset(src, i);

        std::vector<char> raw(old_size);
        if (fseeko(fin, (off_t)(src_data_offset + old_offset), SEEK_SET) != 0) { perror("fseeko in"); return 1; }
        if (fread(raw.data(), 1, old_size, fin) != old_size) { fprintf(stderr, "short read %s\n", name); return 1; }

        ggml_tensor *src_t = ggml_get_tensor(meta_ctx, name);
        int64_t ne0 = src_t->ne[0];
        int64_t nelements = ggml_nelements(src_t);

        std::vector<char> outbuf;
        size_t written;
        if (new_types[i] != old_type) {
            n_changed++;
            std::vector<float> fbuf(nelements);
            const ggml_type_traits *tt_old = ggml_get_type_traits(old_type);
            if (!tt_old->to_float) { fprintf(stderr, "no to_float for %s\n", name); return 1; }
            tt_old->to_float(raw.data(), fbuf.data(), nelements);

            int64_t nrows = nelements / ne0;
            outbuf.resize(ggml_row_size(new_types[i], ne0) * nrows);
            written = ggml_quantize_chunk(new_types[i], fbuf.data(), outbuf.data(), 0, nrows, ne0, nullptr);
            if (written != outbuf.size()) { fprintf(stderr, "size mismatch %s: %zu vs %zu\n", name, written, outbuf.size()); return 1; }
        } else {
            outbuf = raw;
            written = outbuf.size();
        }

        if (fwrite(outbuf.data(), 1, written, fout) != written) { fprintf(stderr, "short write %s\n", name); return 1; }
        size_t padded = GGML_PAD(written, alignment);
        size_t pad = padded - written;
        if (pad > 0) {
            std::vector<char> zeros(pad, 0);
            if (fwrite(zeros.data(), 1, pad, fout) != pad) { fprintf(stderr, "pad write fail %s\n", name); return 1; }
        }
        printf("%-40s %-8s -> %-8s  %10zu -> %10zu bytes\n", name,
               ggml_type_name(old_type), ggml_type_name(new_types[i]), old_size, written);
    }

    fclose(fin);
    fclose(fout);
    gguf_free(src);
    gguf_free(out);
    printf("done: %lld tensors, %d requantized to %s\n", (long long)n_tensors, n_changed, type_name.c_str());
    return 0;
}
