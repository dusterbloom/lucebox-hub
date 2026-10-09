import re

path = "/home/duster/qwen4exp-exact-stack/src/server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"
with open(path) as f:
    src = f.read()

old = """        ggml_type memo_type = GGML_TYPE_COUNT;
        const ggml_tensor * memo_src1 =
            ggml_cuda_hc_upmix_row8_consumer(graph, i + 1 + mix_count, mix.dst, memo_type);
        ggml_cuda_hc_upmix_row8_pair pair {
            i, mix_count, up, mix.xn, mix.dst, memo_src1, memo_type, mix.scale, mix.bias
        };
        if (ggml_cuda_hc_upmix_row8_pair_quick_valid(graph, pair)) {
            plan.pairs.push_back(pair);
        }
    }"""

new = """        ggml_type memo_type = GGML_TYPE_COUNT;
        const ggml_tensor * memo_src1 =
            ggml_cuda_hc_upmix_row8_consumer(graph, i + 1 + mix_count, mix.dst, memo_type);
        ggml_cuda_hc_upmix_row8_pair pair {
            i, mix_count, up, mix.xn, mix.dst, memo_src1, memo_type, mix.scale, mix.bias
        };
        const bool pair_valid = ggml_cuda_hc_upmix_row8_pair_quick_valid(graph, pair);
        if (getenv("LUCE_DEBUG_UPMIX") && (!pair_valid || plan.pairs.size() < 3)) {
            fprintf(stderr,
                "[upmix-debug] site_i=%d mix_count=%d memo_src1=%p memo_type=%s valid=%d\\n",
                i, mix_count, (const void *) memo_src1,
                memo_src1 ? ggml_type_name(memo_type) : "(none)", (int) pair_valid);
            if (memo_src1) {
                fprintf(stderr,
                    "[upmix-debug]   memo_src1: type=%s ne=[%lld,%lld,%lld,%lld] contiguous=%d data=%p buffer=%p\\n",
                    ggml_type_name(memo_src1->type),
                    (long long) memo_src1->ne[0], (long long) memo_src1->ne[1],
                    (long long) memo_src1->ne[2], (long long) memo_src1->ne[3],
                    (int) ggml_is_contiguous(memo_src1), memo_src1->data, memo_src1->buffer);
            }
            fprintf(stderr,
                "[upmix-debug]   mixed: type=%s ne0=%lld nrows=%lld contiguous=%d data=%p scale=%f bias=%f\\n",
                ggml_type_name(mix.dst->type), (long long) mix.dst->ne[0],
                (long long) ggml_nrows(mix.dst), (int) ggml_is_contiguous(mix.dst),
                mix.dst->data, mix.scale, mix.bias);
        }
        if (pair_valid) {
            plan.pairs.push_back(pair);
        }
    }"""

assert old in src, "anchor not found"
src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("patched")
