path = "/home/duster/qwen4exp-exact-stack/src/server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"
with open(path) as f:
    src = f.read()

old = """        const bool pair_valid = ggml_cuda_hc_upmix_row8_pair_quick_valid(graph, pair);
        if (getenv("LUCE_DEBUG_UPMIX") && (!pair_valid || plan.pairs.size() < 3)) {"""

new = """        const bool pair_valid = ggml_cuda_hc_upmix_row8_pair_quick_valid(graph, pair);
        if (getenv("LUCE_DEBUG_UPMIX") && !memo_src1) {
            // consumer() found nothing in its accepted-type window; scan the same
            // range manually and report the first node that actually reads `mixed`,
            // regardless of weight type, so we can see what rejected it.
            for (int j = i + 1 + mix_count; j < graph->n_nodes; ++j) {
                const ggml_tensor * node = graph->nodes[j];
                if ((node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) ||
                    !node->src[0] || !node->src[1]) continue;
                const ggml_tensor * s1 = node->src[1];
                const bool direct = s1 == mix.dst;
                const bool reshape = s1->op == GGML_OP_RESHAPE && s1->src[0] == mix.dst &&
                    s1->data == mix.dst->data;
                if (!direct && !reshape) continue;
                fprintf(stderr,
                    "[upmix-debug]   site_i=%d first-real-consumer: node_j=%d op=%s "
                    "direct=%d reshape=%d weight_type=%s weight_ne=[%lld,%lld] src1_type=%s src1_ne0=%lld\\n",
                    i, j, ggml_op_name(node->op), (int) direct, (int) reshape,
                    ggml_type_name(node->src[0]->type),
                    (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1],
                    ggml_type_name(s1->type), (long long) s1->ne[0]);
                break;
            }
        }
        if (getenv("LUCE_DEBUG_UPMIX") && (!pair_valid || plan.pairs.size() < 3)) {"""

assert old in src, "anchor2 not found"
src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("patched2")
