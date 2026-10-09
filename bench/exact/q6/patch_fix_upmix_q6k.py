path = "/home/duster/qwen4exp-exact-stack/src/server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"
with open(path) as f:
    src = f.read()

# 1) consumer(): the accepted "direct" (non-expert) consumer weight type was
#    hard-coded to Q8_0 only. ggml_cuda_mmvq_set_hc_upmix_row8() (the only place
#    that actually reads this pair) consumes p.xn/p.mixed/p.scale/p.bias -- it
#    never touches the downstream consumer's weight tensor at all. memo_src1 /
#    memo_type exist solely to confirm the mixed buffer is handed off to a real
#    next-layer matmul (an aliasing/safety check), not a computational
#    dependency on that matmul's weight type. Dense Q8_0 tensors feeding this
#    site can now be Q6_K (option A2 requant), so widen the accepted direct
#    type to include it.
old_consumer = """        const ggml_type wt = node->src[0]->type;
        if (node->op == GGML_OP_MUL_MAT_ID && reshape &&
            (wt == GGML_TYPE_Q4_K || wt == GGML_TYPE_Q5_K)) {
            weight_type = wt;
            return src1;
        }
        if (direct && wt == GGML_TYPE_Q8_0 && !q8) {
            q8 = src1;
        }
    }
    if (q8) weight_type = GGML_TYPE_Q8_0;
    return q8;
}"""
new_consumer = """        const ggml_type wt = node->src[0]->type;
        if (node->op == GGML_OP_MUL_MAT_ID && reshape &&
            (wt == GGML_TYPE_Q4_K || wt == GGML_TYPE_Q5_K)) {
            weight_type = wt;
            return src1;
        }
        // Direct (non-expert) consumer: historically Q8_0-only dense weight.
        // The upmix kernel itself never reads this weight (see call site at
        // ggml_cuda_mmvq_set_hc_upmix_row8, which only takes xn/mixed/scale/bias),
        // so this is purely an aliasing/handoff sanity check -- safe to widen to
        // any dense type this consumer can legitimately be (Q8_0 or Q6_K).
        if (direct && (wt == GGML_TYPE_Q8_0 || wt == GGML_TYPE_Q6_K) && !q8) {
            q8 = src1;
            weight_type = wt;
        }
    }
    return q8;
}"""
assert old_consumer in src, "consumer anchor not found"
src = src.replace(old_consumer, new_consumer, 1)

# 2) quick_valid(): widen the accepted memo_src0_type set to match.
old_valid = """        (pair.memo_src0_type == GGML_TYPE_Q8_0 ||
         pair.memo_src0_type == GGML_TYPE_Q4_K || pair.memo_src0_type == GGML_TYPE_Q5_K);"""
new_valid = """        (pair.memo_src0_type == GGML_TYPE_Q8_0 ||
         pair.memo_src0_type == GGML_TYPE_Q4_K || pair.memo_src0_type == GGML_TYPE_Q5_K ||
         pair.memo_src0_type == GGML_TYPE_Q6_K);"""
assert old_valid in src, "quick_valid anchor not found"
src = src.replace(old_valid, new_valid, 1)

with open(path, "w") as f:
    f.write(src)
print("patched fix")
