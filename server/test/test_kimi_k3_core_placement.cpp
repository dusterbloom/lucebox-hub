#include "kimi_k3/kimi_k3_backend.h"

#include <cstdio>
#include <cstdlib>

using namespace dflash::common;

#define REQUIRE(condition) do {                                           \
    if (!(condition)) {                                                   \
        std::fprintf(stderr, "requirement failed at %s:%d: %s\n",       \
                     __FILE__, __LINE__, #condition);                     \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

int main() {
    KimiK3CorePlacement placement = KimiK3CorePlacement::Accelerator;
    REQUIRE(parse_kimi_k3_core_placement("cpu", placement));
    REQUIRE(placement == KimiK3CorePlacement::Cpu);
    REQUIRE(std::string(kimi_k3_core_placement_name(placement)) == "cpu");
    REQUIRE(parse_kimi_k3_core_placement("accelerator", placement));
    REQUIRE(placement == KimiK3CorePlacement::Accelerator);
    REQUIRE(!parse_kimi_k3_core_placement("gpu", placement));

    int prefill_chunk = 0;
    REQUIRE(parse_kimi_k3_prefill_chunk(nullptr, prefill_chunk));
    REQUIRE(prefill_chunk == 1);
    REQUIRE(parse_kimi_k3_prefill_chunk("", prefill_chunk));
    REQUIRE(prefill_chunk == 1);
    REQUIRE(parse_kimi_k3_prefill_chunk("1", prefill_chunk));
    REQUIRE(prefill_chunk == 1);
    REQUIRE(parse_kimi_k3_prefill_chunk("2", prefill_chunk));
    REQUIRE(prefill_chunk == 2);
    REQUIRE(parse_kimi_k3_prefill_chunk("4", prefill_chunk));
    REQUIRE(prefill_chunk == 4);
    REQUIRE(!parse_kimi_k3_prefill_chunk("0", prefill_chunk));
    REQUIRE(!parse_kimi_k3_prefill_chunk("8", prefill_chunk));
    REQUIRE(!parse_kimi_k3_prefill_chunk("2x", prefill_chunk));

    std::string error;
    ggml_backend_t cpu = init_kimi_k3_core_backend(
        KimiK3CorePlacement::Cpu, 0, &error);
    REQUIRE(cpu != nullptr);
    ggml_backend_free(cpu);

    std::printf("Kimi K3 core placement test passed\n");
    return 0;
}
