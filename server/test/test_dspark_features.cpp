#include "draft/dspark_features.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace dflash::common;

namespace {

bool check(bool condition, const char * expression, int line) {
    if (condition) return true;
    std::fprintf(stderr, "check failed at line %d: %s\n", line, expression);
    return false;
}

}  // namespace

#define CHECK(expr) do { if (!check((expr), #expr, __LINE__)) return 1; } while (false)

int main() {
    std::vector<float> features;
    CHECK(make_dspark_log_snr_features(4, -9.0f, 9.0f, features));
    CHECK(features.size() == 4u * 128u);

    // t=0 => sin=0 and cos=1 for every frequency on masked rows.
    for (int row = 1; row < 4; ++row) {
        for (int i = 0; i < 64; ++i) {
            CHECK(features[(size_t)row * 128u + (size_t)i] == 0.0f);
            CHECK(features[(size_t)row * 128u + 64u + (size_t)i] == 1.0f);
        }
    }

    // The clean anchor uses t=1000 and differs from the masked rows.
    CHECK(std::fabs(features[0]) > 0.1f);
    CHECK(std::fabs(features[64] - 1.0f) > 0.1f);

    CHECK(!make_dspark_log_snr_features(0, -9.0f, 9.0f, features));
    CHECK(!make_dspark_log_snr_features(4, 9.0f, -9.0f, features));
    return 0;
}
