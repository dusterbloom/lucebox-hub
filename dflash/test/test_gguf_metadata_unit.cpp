// test_gguf_metadata_unit.cpp — T1 unit test for common/gguf_metadata.h.
//
// Exercises the shared metadata helpers without a GGUF file on disk:
// builds an in-memory gguf_context via gguf_init_empty + gguf_set_val_*,
// then asserts the read paths. CHECK()-based per dflash/docs/TEST_TIERS.md
// — bare assert() would be stripped under NDEBUG and silently pass.

#include "common/gguf_metadata.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "%s:%d CHECK(%s) FAILED\n",                     \
                     __FILE__, __LINE__, #cond);                             \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

using namespace dflash::common;

namespace {

struct GgufFixture {
    struct gguf_context * ctx = nullptr;
    GgufFixture() {
        ctx = gguf_init_empty();
        gguf_set_val_str(ctx, "general.architecture", "qwen35");
        gguf_set_val_str(ctx, "general.name",         "Qwen3.6-27B");
        gguf_set_val_u32(ctx, "qwen35.block_count",       64);
        gguf_set_val_u32(ctx, "qwen35.embedding_length",  5120);
        gguf_set_val_u32(ctx, "qwen35.nextn_predict_layers", 2);
    }
    ~GgufFixture() { if (ctx) gguf_free(ctx); }
};

void test_get_u32_or() {
    GgufFixture f;
    CHECK(gguf_get_u32_or(f.ctx, "qwen35.block_count",      0)    == 64);
    CHECK(gguf_get_u32_or(f.ctx, "qwen35.embedding_length", 0)    == 5120);
    // Absent key returns the default.
    CHECK(gguf_get_u32_or(f.ctx, "qwen35.absent_key",       999)  == 999);
    CHECK(gguf_get_u32_or(f.ctx, "qwen35.absent_key",       0)    == 0);
    std::printf("[gguf_metadata] get_u32_or OK\n");
}

void test_get_str_or() {
    GgufFixture f;
    CHECK(gguf_get_str_or(f.ctx, "general.architecture", "x") == "qwen35");
    CHECK(gguf_get_str_or(f.ctx, "general.name",         "x") == "Qwen3.6-27B");
    CHECK(gguf_get_str_or(f.ctx, "absent.key", "fallback")    == "fallback");
    std::printf("[gguf_metadata] get_str_or OK\n");
}

void test_require_u32() {
    GgufFixture f;
    uint32_t v = 0;
    std::string err;

    CHECK(gguf_require_u32(f.ctx, "qwen35.block_count", v, err));
    CHECK(v == 64);
    CHECK(err.empty());

    v = 12345;
    CHECK(!gguf_require_u32(f.ctx, "absent.key", v, err));
    CHECK(!err.empty());
    CHECK(err.find("absent.key") != std::string::npos);
    // On failure, out is not mutated.
    CHECK(v == 12345);
    std::printf("[gguf_metadata] require_u32 OK\n");
}

void test_require_str() {
    GgufFixture f;
    std::string v;
    std::string err;

    CHECK(gguf_require_str(f.ctx, "general.architecture", v, err));
    CHECK(v == "qwen35");
    CHECK(err.empty());

    CHECK(!gguf_require_str(f.ctx, "absent.key", v, err));
    CHECK(!err.empty());
    CHECK(err.find("absent.key") != std::string::npos);
    std::printf("[gguf_metadata] require_str OK\n");
}

void test_check_architecture_match() {
    GgufFixture f;
    std::string err;
    CHECK(gguf_check_architecture(f.ctx, "qwen35", err));
    CHECK(err.empty());
    std::printf("[gguf_metadata] check_architecture match OK\n");
}

void test_check_architecture_mismatch() {
    GgufFixture f;
    std::string err;
    CHECK(!gguf_check_architecture(f.ctx, "gemma4", err));
    CHECK(!err.empty());
    CHECK(err.find("qwen35")  != std::string::npos);
    CHECK(err.find("gemma4")  != std::string::npos);
    std::printf("[gguf_metadata] check_architecture mismatch OK\n");
}

void test_check_architecture_missing() {
    // Build a fixture with no general.architecture key.
    struct gguf_context * ctx = gguf_init_empty();
    gguf_set_val_u32(ctx, "some.other.key", 1);
    std::string err;
    CHECK(!gguf_check_architecture(ctx, "qwen35", err));
    CHECK(err.find("general.architecture") != std::string::npos);
    gguf_free(ctx);
    std::printf("[gguf_metadata] check_architecture missing OK\n");
}

} // namespace

int main() {
    test_get_u32_or();
    test_get_str_or();
    test_require_u32();
    test_require_str();
    test_check_architecture_match();
    test_check_architecture_mismatch();
    test_check_architecture_missing();
    std::printf("[gguf_metadata] all unit tests PASS\n");
    return 0;
}
