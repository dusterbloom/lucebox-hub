#include "kimi_k3_panel_fit.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dflash::common {

void KimiK3DiagonalStats::reset(size_t dimension) {
    s0 = 0.0;
    observations = 0;
    sx.assign(dimension, 0.0);
    sxx.assign(dimension, 0.0);
    sy.assign(dimension, 0.0);
    sxy.assign(dimension, 0.0);
}

bool KimiK3DiagonalStats::observe(
        const float * input, const float * output,
        size_t dimension, double weight, std::string * error) {
    if (!input || !output || dimension == 0 || dimension != sx.size() ||
        sxx.size() != dimension || sy.size() != dimension ||
        sxy.size() != dimension || !std::isfinite(weight) || weight < 0.0) {
        if (error) *error = "invalid diagonal sufficient-statistic observation";
        return false;
    }
    if (weight == 0.0) return true;
    for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
        const double x = input[coordinate];
        const double y = output[coordinate];
        if (!std::isfinite(x) || !std::isfinite(y)) {
            if (error) *error = "non-finite diagonal fit observation";
            return false;
        }
        sx[coordinate] += weight * x;
        sxx[coordinate] += weight * x * x;
        sy[coordinate] += weight * y;
        sxy[coordinate] += weight * x * y;
    }
    s0 += weight;
    ++observations;
    return true;
}

bool fit_kimi_k3_diagonal_panel(
        const KimiK3DiagonalStats & stats,
        KimiK3DiagonalPanel & panel,
        std::string * error) {
    panel = KimiK3DiagonalPanel{};
    const size_t dimension = stats.dimension();
    if (dimension == 0 || stats.sxx.size() != dimension ||
        stats.sy.size() != dimension || stats.sxy.size() != dimension ||
        !std::isfinite(stats.s0) || stats.s0 <= 0.0) {
        if (error) *error = "diagonal fit has no valid observations";
        return false;
    }

    panel.offset.resize(dimension);
    panel.gain.resize(dimension);
    for (size_t coordinate = 0; coordinate < dimension; ++coordinate) {
        const double sx = stats.sx[coordinate];
        const double sxx = stats.sxx[coordinate];
        const double sy = stats.sy[coordinate];
        const double sxy = stats.sxy[coordinate];
        const double denominator = stats.s0 * sxx - sx * sx;
        const double scale = std::max(
            1.0, std::fabs(stats.s0 * sxx) + std::fabs(sx * sx));
        double gain = 0.0;
        if (std::isfinite(denominator) &&
            std::fabs(denominator) >
                64.0 * std::numeric_limits<double>::epsilon() * scale) {
            gain = (stats.s0 * sxy - sx * sy) / denominator;
        } else {
            ++panel.degenerate_coordinates;
        }
        const double offset = (sy - gain * sx) / stats.s0;
        if (!std::isfinite(gain) || !std::isfinite(offset) ||
            gain > std::numeric_limits<float>::max() ||
            gain < -std::numeric_limits<float>::max() ||
            offset > std::numeric_limits<float>::max() ||
            offset < -std::numeric_limits<float>::max()) {
            if (error) *error = "diagonal fit produced a non-finite parameter";
            panel = KimiK3DiagonalPanel{};
            return false;
        }
        panel.gain[coordinate] = static_cast<float>(gain);
        panel.offset[coordinate] = static_cast<float>(offset);
    }
    return true;
}

} // namespace dflash::common
