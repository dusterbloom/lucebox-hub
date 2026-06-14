// CUDA sampling kernel for diffusion denoising.
//
// Performs softmax + multinomial sampling + Shannon entropy entirely on the
// GPU, given the [n_vocab, C] F32 logit tensor that is already device-resident
// after the ggml forward pass.  Only the small result arrays (sampled[C],
// entropy[C], argmax[C]) — ~3 KB — are copied to host; the 268 MB logit
// tensor never leaves the GPU.
//
// Math matches the reference (diffusion.cpp:678-708) exactly:
//   temp_inv = 1/t
//   m        = max(row * temp_inv)
//   Z        = sum exp(row[v] * temp_inv - m)
//   target   = u[pos] * Z         (u drawn by caller on host, seeded reproducibly)
//   sampled  = first v where cumulative_exp >= target
//   H        = -sum (exp(v*tinv-m)/Z) * log(exp(v*tinv-m)/Z)
//   argmax   = argmax(row)

#pragma once
#ifdef DFLASH27B_BACKEND_CUDA

#include <cstdint>
#include <cuda_runtime.h>

namespace dflash::diffusion {

// Launch the sampling kernel.
//
// logits_dev:   device pointer to [n_vocab, C] F32 logits, row-major
//               (row c spans logits_dev[c*n_vocab .. c*n_vocab+n_vocab-1]).
//               Must be valid CUDA device memory; caller guarantees it is
//               fully written (ggml_backend_graph_compute has returned).
// u_dev:        device pointer to C float uniforms in [0,1).
// temp_inv:     1/temperature (scalar, same for all positions this step).
// C:            number of canvas positions (block count).
// n_vocab:      vocabulary size (262144 for Gemma).
// sampled_dev:  device output, int32_t[C] — multinomial-sampled token IDs.
// entropy_dev:  device output, float[C]   — Shannon entropy per position.
// argmax_dev:   device output, int32_t[C] — greedy argmax per position.
// stream:       CUDA stream; pass 0 for default stream.
void diffusion_sample_gpu(
    const float *  logits_dev,
    const float *  u_dev,
    float          temp_inv,
    int            C,
    int            n_vocab,
    int32_t *      sampled_dev,
    float *        entropy_dev,
    int32_t *      argmax_dev,
    cudaStream_t   stream = 0);

}  // namespace dflash::diffusion

#endif  // DFLASH27B_BACKEND_CUDA
