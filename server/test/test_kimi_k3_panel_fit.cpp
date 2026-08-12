#include "kimi_k3/kimi_k3_panel_fit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace dflash::common;

#define REQUIRE(condition) do {                                           \
    if (!(condition)) {                                                   \
        std::fprintf(stderr, "requirement failed at %s:%d: %s\n",       \
                     __FILE__, __LINE__, #condition);                     \
        std::exit(1);                                                     \
    }                                                                     \
} while (0)

int main() {
    constexpr size_t dimension = 7;
    const std::vector<float> expected_offset = {
        -0.3f, 0.1f, 0.5f, -0.7f, 0.9f, 0.2f, -0.4f};
    const std::vector<float> expected_gain = {
        0.2f, -0.4f, 0.7f, 1.1f, -0.8f, 0.05f, 0.6f};
    KimiK3DiagonalStats stats;
    stats.reset(dimension);
    std::string error;
    for (int sample = 0; sample < 64; ++sample) {
        std::vector<float> input(dimension);
        std::vector<float> output(dimension);
        for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
            input[coordinate] =
                (static_cast<float>((sample + 3 * coordinate) % 19) - 9.0f) /
                (3.0f + static_cast<float>(coordinate));
            output[coordinate] = expected_offset[coordinate] +
                expected_gain[coordinate] * input[coordinate];
        }
        const double weight = 0.25 + (sample % 11) * 0.1;
        REQUIRE(stats.observe(input.data(), output.data(), dimension,
                              weight, &error));
    }
    KimiK3DiagonalPanel panel;
    REQUIRE(fit_kimi_k3_diagonal_panel(stats, panel, &error));
    REQUIRE(panel.degenerate_coordinates == 0);
    for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
        REQUIRE(std::fabs(panel.offset[coordinate] -
                          expected_offset[coordinate]) < 2.0e-6f);
        REQUIRE(std::fabs(panel.gain[coordinate] -
                          expected_gain[coordinate]) < 2.0e-6f);
    }

    KimiK3DiagonalStats degenerate;
    degenerate.reset(1);
    const float fixed_input = 2.0f;
    const float fixed_output = 5.0f;
    REQUIRE(degenerate.observe(
        &fixed_input, &fixed_output, 1, 1.0, &error));
    REQUIRE(degenerate.observe(
        &fixed_input, &fixed_output, 1, 1.0, &error));
    REQUIRE(fit_kimi_k3_diagonal_panel(degenerate, panel, &error));
    REQUIRE(panel.degenerate_coordinates == 1);
    REQUIRE(panel.gain[0] == 0.0f);
    REQUIRE(std::fabs(panel.offset[0] - 5.0f) < 1.0e-6f);

    std::printf("Kimi K3 diagonal panel fit test passed\n");
    return 0;
}
