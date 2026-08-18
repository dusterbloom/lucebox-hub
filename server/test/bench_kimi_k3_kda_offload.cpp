#include "dflash27b.h"
#include "kimi_k3/kimi_k3_internal.h"

#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 7) {
        std::fprintf(stderr,
            "usage: %s MODEL [layer=88] [iterations=7] "
            "[gpu=0|-1:cpu-copy] [output.json] [threads=12]\n",
            argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    const int layer = argc > 2 ? std::atoi(argv[2]) : 88;
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 7;
    const int gpu = argc > 4 ? std::atoi(argv[4]) : 0;
    const std::string output = argc > 5 ? argv[5] : "";
    const int threads = argc > 6 ? std::atoi(argv[6]) : 12;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t accelerator = gpu < 0
        ? ggml_backend_cpu_init() : ggml_backend_cuda_init(gpu);
    if (!cpu || !accelerator) {
        std::fprintf(stderr, "backend initialization failed\n");
        if (accelerator) ggml_backend_free(accelerator);
        if (cpu) ggml_backend_free(cpu);
        return 1;
    }
    ggml_backend_cpu_set_n_threads(cpu, threads);
    if (gpu < 0) ggml_backend_cpu_set_n_threads(accelerator, threads);
    KimiK3Weights weights;
    KimiK3LoadOptions options;
    options.stream_routed_experts = true;
    options.mmap_resident_tensors = true;
    if (!load_kimi_k3_gguf(model, cpu, weights, options)) {
        std::fprintf(stderr, "model load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(accelerator);
        ggml_backend_free(cpu);
        return 1;
    }

    KimiK3KdaLayerBenchmarkResult result;
    std::string error;
    const bool ok = benchmark_kimi_k3_kda_layer(
        cpu, accelerator, weights, layer, iterations, result, &error);
    if (!ok) {
        std::fprintf(stderr, "KDA benchmark failed: %s\n", error.c_str());
        free_kimi_k3_weights(weights);
        ggml_backend_free(accelerator);
        ggml_backend_free(cpu);
        return 1;
    }
    std::ostringstream json;
    json << std::setprecision(12)
         << "{\n"
         << "  \"schema\": \"k3_p34_isolated_kda_offload_v1\",\n"
         << "  \"model\": \"" << model << "\",\n"
         << "  \"model_layer\": " << result.model_layer << ",\n"
         << "  \"iterations\": " << result.iterations << ",\n"
         << "  \"destination\": \""
         << (gpu < 0 ? "cpu-copy" : "cuda") << "\",\n"
         << "  \"cpu_threads\": " << threads << ",\n"
         << "  \"weight_bytes\": " << result.weight_bytes << ",\n"
         << "  \"cpu_median_ms\": " << result.cpu_median_ms << ",\n"
         << "  \"accelerator_median_ms\": "
         << result.accelerator_median_ms << ",\n"
         << "  \"speedup\": " << result.speedup << ",\n"
         << "  \"relative_l2\": " << result.relative_l2 << ",\n"
         << "  \"cosine\": " << result.cosine << ",\n"
         << "  \"max_abs\": " << result.max_abs << "\n"
         << "}\n";
    std::printf("%s", json.str().c_str());
    if (!output.empty()) {
        std::ofstream file(output);
        if (!file) {
            std::fprintf(stderr, "cannot open output: %s\n", output.c_str());
            free_kimi_k3_weights(weights);
            ggml_backend_free(accelerator);
            ggml_backend_free(cpu);
            return 1;
        }
        file << json.str();
    }
    free_kimi_k3_weights(weights);
    ggml_backend_free(accelerator);
    ggml_backend_free(cpu);
    return 0;
}
