#include "kimi_k3/kimi_k3_internal.h"

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
    constexpr int stop_layer = 1;
    REQUIRE(kimi_k3_capture_tensor_required(
        "token_embd.weight", stop_layer));
    REQUIRE(!kimi_k3_capture_tensor_required(
        "output.weight", stop_layer));
    REQUIRE(!kimi_k3_capture_tensor_required(
        "output_norm.weight", stop_layer));

    // The complete exact prefix is resident.
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.0.attn_q.weight", stop_layer));
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.0.ffn_gate.weight", stop_layer));

    // The stop layer retains attention, router, latent down-projection, and
    // the small routed join needed for held-out downstream measurements.
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.1.attn_q.weight", stop_layer));
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.1.ffn_gate_inp.weight", stop_layer));
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.1.exp_probs_b.bias", stop_layer));
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.1.ffn_routed_down.weight", stop_layer));
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.1.ffn_routed_norm.weight", stop_layer));
    REQUIRE(kimi_k3_capture_tensor_required(
        "blk.1.ffn_routed_up.weight", stop_layer));

    // The large shared expert, routed bank, and all later layers are absent.
    REQUIRE(!kimi_k3_capture_tensor_required(
        "blk.1.ffn_gate_shexp.weight", stop_layer));
    REQUIRE(!kimi_k3_capture_tensor_required(
        "blk.1.ffn_up_shexp.weight", stop_layer));
    REQUIRE(!kimi_k3_capture_tensor_required(
        "blk.1.ffn_down_shexp.weight", stop_layer));
    REQUIRE(!kimi_k3_capture_tensor_required(
        "blk.1.ffn_gate_exps.weight", stop_layer));
    REQUIRE(!kimi_k3_capture_tensor_required(
        "blk.2.attn_q.weight", stop_layer));

    std::printf("Kimi K3 selective-load policy test passed\n");
    return 0;
}
