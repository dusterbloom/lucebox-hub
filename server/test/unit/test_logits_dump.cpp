// C1.3-05: the logits-dump hook and teacher forcing for the logit-equivalence
// gate. Both are env-driven and must be OFF unless the env names a file.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/logits_dump.h"

#include <cstdio>
#include <vector>

using namespace luce::common;

namespace {
struct LogitsDumpFixture {};

std::vector<char> slurp(const char * p) {
    std::vector<char> v;
    if (FILE * f = std::fopen(p, "rb")) {
        char b[256];
        size_t n;
        while ((n = std::fread(b, 1, sizeof b, f)) > 0) v.insert(v.end(), b, b + n);
        std::fclose(f);
    }
    return v;
}
}  // namespace

TEST_CASE(LogitsDumpFixture, dump_is_off_when_the_env_is_unset_or_empty) {
    CHECK_FALSE(LogitsDump::from_env(nullptr).enabled());
    CHECK_FALSE(LogitsDump::from_env("").enabled());
    const float v[2] = {1.f, 2.f};
    CHECK_FALSE(LogitsDump::from_env(nullptr).append(v, 2));
}

TEST_CASE(LogitsDumpFixture, dump_appends_one_record_per_call_in_order) {
    const char * p = "logits_dump_test.bin";
    std::remove(p);
    const LogitsDump d = LogitsDump::from_env(p);
    CHECK(d.enabled());
    const float a[3] = {1.f, 2.f, 3.f}, b[3] = {4.f, 5.f, 6.f};
    CHECK(d.append(a, 3));
    CHECK(d.append(b, 3));
    const std::vector<char> raw = slurp(p);
    std::remove(p);
    REQUIRE(raw.size() == 6 * sizeof(float));
    const float * f = reinterpret_cast<const float *>(raw.data());
    CHECK(f[0] == 1.f);
    CHECK(f[2] == 3.f);
    CHECK(f[3] == 4.f);
    CHECK(f[5] == 6.f);
}

TEST_CASE(LogitsDumpFixture, teacher_force_is_off_when_unset_and_empty_for_a_missing_file) {
    CHECK_FALSE(TeacherForce::from_env(nullptr).enabled());
    CHECK_FALSE(TeacherForce::from_env("").enabled());
    CHECK_FALSE(TeacherForce::from_env("no_such_teacher_file.bin").enabled());
}

TEST_CASE(LogitsDumpFixture, teacher_force_serves_the_file_tokens_by_step_and_declines_past_the_end) {
    const char * p = "teacher_force_test.bin";
    const int32_t t[3] = {7, 8, 9};
    FILE * f = std::fopen(p, "wb");
    std::fwrite(t, sizeof t[0], 3, f);
    std::fclose(f);
    const TeacherForce tf = TeacherForce::from_env(p);
    std::remove(p);
    REQUIRE(tf.enabled());
    CHECK(tf.at(0) == 7);
    CHECK(tf.at(2) == 9);
    CHECK(tf.at(3) == -1);
}
