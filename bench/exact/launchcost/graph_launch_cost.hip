// Microbench: per-node dispatch cost inside a replayed HIP graph on gfx1151.
// Builds a single-stream graph of N kernel nodes, replays it R times, reports us/node.
//
// Three node shapes, selected by --shape:
//   empty : <<<1,32>>> kernel that does nothing (pure launch/dispatch floor)
//   chain : <<<40,256>>> kernel, each node reads the 4 KB its predecessor wrote (true
//           dependency chain, forces the command processor to honor node ordering)
//   chain1m: same dependency chain, but each node touches 1 MB (bandwidth-visible node,
//           to separate "dispatch tax" from "this node's own memory traffic")
//
// Usage:
//   graph_launch_cost <N nodes: 256|1024|2048> <shape: empty|chain|chain1m> [replays=200]
//
// Build:
//   hipcc -O3 --offload-arch=gfx1151 -o graph_launch_cost graph_launch_cost.hip

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>

#define HIP_CHECK(x)                                                                   \
  do {                                                                                 \
    hipError_t _e = (x);                                                              \
    if (_e != hipSuccess) {                                                           \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

// Pure no-op node: smallest possible kernel, isolates launch/dispatch cost only.
__global__ void k_empty() {}

// Dependency-chain node: reads 1024 floats (4 KB) written by the previous node, writes its own
// 1024 floats. Forces true data dependency so the command processor cannot reorder/overlap nodes.
__global__ void k_chain4k(const float* __restrict__ in, float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 1024) out[i] = in[i] + 1.0f;
}

// Same dependency chain, 1 MB per node (262144 floats) instead of 4 KB — makes the node's own
// memory traffic visible so we can separate dispatch tax from real work.
__global__ void k_chain1m(const float* __restrict__ in, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + 1.0f;
}

static double now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <N nodes> <shape: empty|chain|chain1m> [replays=200]\n", argv[0]);
    return 1;
  }
  int n_nodes = atoi(argv[1]);
  std::string shape = argv[2];
  int replays = argc > 3 ? atoi(argv[3]) : 200;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Buffers for chain shapes: ping-pong 4KB or 1MB buffers, one pair per node to keep the
  // dependency explicit and avoid aliasing hazards that could make the graph capture collapse
  // independent nodes.
  const int chain4k_elems = 1024;
  const int chain1m_elems = 262144;  // 1 MB of float32

  std::vector<float*> bufs;
  if (shape == "chain") {
    bufs.resize(n_nodes + 1);
    for (int i = 0; i <= n_nodes; i++) {
      HIP_CHECK(hipMalloc(&bufs[i], chain4k_elems * sizeof(float)));
      HIP_CHECK(hipMemsetAsync(bufs[i], 0, chain4k_elems * sizeof(float), stream));
    }
  } else if (shape == "chain1m") {
    bufs.resize(n_nodes + 1);
    for (int i = 0; i <= n_nodes; i++) {
      HIP_CHECK(hipMalloc(&bufs[i], chain1m_elems * sizeof(float)));
      HIP_CHECK(hipMemsetAsync(bufs[i], 0, chain1m_elems * sizeof(float), stream));
    }
  } else if (shape != "empty") {
    fprintf(stderr, "unknown shape %s\n", shape.c_str());
    return 1;
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  // Capture the graph.
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int i = 0; i < n_nodes; i++) {
    if (shape == "empty") {
      k_empty<<<1, 32, 0, stream>>>();
    } else if (shape == "chain") {
      k_chain4k<<<4, 256, 0, stream>>>(bufs[i], bufs[i + 1]);
    } else {  // chain1m
      int threads = 256;
      int blocks = (chain1m_elems + threads - 1) / threads;
      k_chain1m<<<blocks, threads, 0, stream>>>(bufs[i], bufs[i + 1], chain1m_elems);
    }
  }
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t graph_exec;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  // Warmup replays (not timed) — let clocks/caches settle.
  for (int w = 0; w < 10; w++) {
    HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  // Timed replays.
  std::vector<double> us_per_replay(replays);
  for (int r = 0; r < replays; r++) {
    double t0 = now_us();
    HIP_CHECK(hipGraphLaunch(graph_exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    double t1 = now_us();
    us_per_replay[r] = t1 - t0;
  }

  // Median.
  std::vector<double> sorted = us_per_replay;
  std::sort(sorted.begin(), sorted.end());
  double median = sorted[sorted.size() / 2];
  double sum = 0;
  for (double v : sorted) sum += v;
  double mean = sum / sorted.size();
  double best = sorted.front();

  printf("shape=%s n_nodes=%d replays=%d  total_us: median=%.3f mean=%.3f best=%.3f  "
         "per_node_us: median=%.4f mean=%.4f best=%.4f\n",
         shape.c_str(), n_nodes, replays, median, mean, best, median / n_nodes, mean / n_nodes,
         best / n_nodes);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  for (float* b : bufs) { HIP_CHECK(hipFree(b)); }
  HIP_CHECK(hipStreamDestroy(stream));
  return 0;
}
