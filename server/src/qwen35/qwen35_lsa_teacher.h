#pragma once

#include "qwen35_lsa_capture.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>

namespace dflash::common {

// Isolated graph state for offline LSA teacher extraction. This deliberately
// does not share StepGraph or any production decode/verify path.
struct Qwen35LsaTeacherStep {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t alloc = nullptr;

    ggml_tensor * inp_embed = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * attn_mask = nullptr;

    QwenGraphOutputs outputs;
    Qwen35LsaCaptureConfig capture_config;
    int kv_start = 0;
    int n_tokens = 0;
    int kq_stride_pad = 32;
};

bool build_qwen35_lsa_teacher_step(
    Qwen35LsaTeacherStep & step,
    const TargetWeights & weights,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens,
    const Qwen35LsaCaptureConfig & config,
    int kq_stride_pad,
    std::string & error);

bool execute_qwen35_lsa_teacher_step(
    Qwen35LsaTeacherStep & step,
    const TargetWeights & weights,
    TargetCache & cache,
    ggml_backend_t backend,
    const int32_t * token_ids,
    Qwen35LsaCaptureBatch & batch,
    std::string & error);

void destroy_qwen35_lsa_teacher_step(Qwen35LsaTeacherStep & step);

}  // namespace dflash::common
