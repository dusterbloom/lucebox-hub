#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

struct KimiK3DiagonalStats {
    double s0 = 0.0;
    uint64_t observations = 0;
    std::vector<double> sx;
    std::vector<double> sxx;
    std::vector<double> sy;
    std::vector<double> sxy;

    void reset(size_t dimension);
    bool observe(const float * input, const float * output,
                 size_t dimension, double weight = 1.0,
                 std::string * error = nullptr);
    size_t dimension() const { return sx.size(); }
};

struct KimiK3DiagonalPanel {
    std::vector<float> offset;
    std::vector<float> gain;
    uint64_t degenerate_coordinates = 0;
};

bool fit_kimi_k3_diagonal_panel(
    const KimiK3DiagonalStats & stats,
    KimiK3DiagonalPanel & panel,
    std::string * error = nullptr);

} // namespace dflash::common
