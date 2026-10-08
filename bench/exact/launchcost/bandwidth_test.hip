// Measure achievable GPU read bandwidth on gfx1151 (Strix Halo APU).
// Grid-stride 128-bit (float4) loads over a 2 GB device buffer, summed into one atomic.
// Median of 20 timed runs. Build: hipcc -O3 --offload-arch=gfx1151 -o bandwidth_test bandwidth_test.hip

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>

#define HIP_CHECK(x)                                                                   \
  do {                                                                                 \
    hipError_t _e = (x);                                                              \
    if (_e != hipSuccess) {                                                           \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

__global__ void k_read_bw(const float4* __restrict__ buf, unsigned long long n4,
                           float* __restrict__ out_sum) {
  unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
  unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
  float acc = 0.0f;
  for (unsigned long long i = idx; i < n4; i += stride) {
    float4 v = buf[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc != 0.0f || idx == 0) {
    atomicAdd(out_sum, acc);
  }
}

static double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  size_t bytes = 2ULL * 1024 * 1024 * 1024;  // 2 GB
  int runs = argc > 1 ? atoi(argv[1]) : 20;

  float4* d_buf;
  float* d_sum;
  unsigned long long n4 = bytes / sizeof(float4);
  HIP_CHECK(hipMalloc(&d_buf, bytes));
  HIP_CHECK(hipMalloc(&d_sum, sizeof(float)));
  HIP_CHECK(hipMemset(d_buf, 1, bytes));  // non-zero pattern, avoids any all-zero shortcut
  HIP_CHECK(hipMemset(d_sum, 0, sizeof(float)));

  int threads = 256;
  int blocks = 4096;  // enough to saturate, grid-stride handles any leftover

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Warmup.
  for (int w = 0; w < 3; w++) {
    k_read_bw<<<blocks, threads, 0, stream>>>(d_buf, n4, d_sum);
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<double> us(runs);
  for (int r = 0; r < runs; r++) {
    double t0 = now_us();
    k_read_bw<<<blocks, threads, 0, stream>>>(d_buf, n4, d_sum);
    HIP_CHECK(hipStreamSynchronize(stream));
    double t1 = now_us();
    us[r] = t1 - t0;
  }

  std::vector<double> sorted = us;
  std::sort(sorted.begin(), sorted.end());
  double median_us = sorted[sorted.size() / 2];
  double best_us = sorted.front();
  double sum = 0;
  for (double v : sorted) sum += v;
  double mean_us = sum / sorted.size();

  double gb = bytes / 1e9;
  printf("bytes=%zu (%.3f GB) runs=%d  us: median=%.1f mean=%.1f best=%.1f  "
         "GB/s: median=%.2f mean=%.2f best=%.2f\n",
         bytes, gb, runs, median_us, mean_us, best_us, gb / (median_us / 1e6),
         gb / (mean_us / 1e6), gb / (best_us / 1e6));

  HIP_CHECK(hipFree(d_buf));
  HIP_CHECK(hipFree(d_sum));
  HIP_CHECK(hipStreamDestroy(stream));
  return 0;
}
