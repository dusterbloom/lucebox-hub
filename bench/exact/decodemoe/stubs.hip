// Host stubs so mmvq.cu links into the microbench without the rest of ggml-hip.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "common.cuh"
#include "quantize.cuh"
#include <cstdarg>

void quantize_row_q8_1_cuda(const float *, const int32_t *, void *, ggml_type, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t) { abort(); }
void quantize_hc_lo_q8_1_cuda(const float *, float *, void *, bool, float, float, cudaStream_t) { abort(); }
extern "C" [[noreturn]] void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt); fprintf(stderr, "%s:%d: ", file, line); vfprintf(stderr, fmt, ap); va_end(ap); fputc('\n', stderr); abort();
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = []() {
        ggml_cuda_device_info i{};
        i.device_count = 1;
        i.devices[0].cc = GGML_CUDA_CC_OFFSET_AMD + 0x1151;
        i.devices[0].warp_size = 32;
        i.devices[0].nsm = 40;
        i.devices[0].smpb = 64 * 1024;
        return i;
    }();
    return info;
}
int ggml_cuda_get_device() { return 0; }
void ggml_cuda_set_device(int) {}
bool ggml_cuda_qwen4exp_enabled() { return true; }
bool ggml_cuda_mmvq_batch_invariant() { return false; }
[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  %s at %s:%d: %s\n", msg, stmt, file, line, func); abort();
}
std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) { abort(); }

extern "C" {
int64_t ggml_blck_size(enum ggml_type t) { return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ? 256 : (t == GGML_TYPE_Q8_1 ? 32 : 1); }
size_t  ggml_type_size(enum ggml_type t) { return t == GGML_TYPE_Q4_K ? 144 : t == GGML_TYPE_Q5_K ? 176 : t == GGML_TYPE_Q8_1 ? 36 : 4; }
const char * ggml_type_name(enum ggml_type) { return "?"; }
size_t  ggml_nbytes(const struct ggml_tensor *) { abort(); }
int64_t ggml_nelements(const struct ggml_tensor *) { abort(); }
int64_t ggml_nrows(const struct ggml_tensor *) { abort(); }
bool ggml_is_contiguous(const struct ggml_tensor *) { abort(); }
bool ggml_is_contiguously_allocated(const struct ggml_tensor *) { abort(); }
bool ggml_are_same_stride(const struct ggml_tensor *, const struct ggml_tensor *) { abort(); }
size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t, const struct ggml_tensor *) { abort(); }
ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t) { abort(); }
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t) { abort(); }
ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int) { abort(); }
}
