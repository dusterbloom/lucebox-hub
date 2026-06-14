// TDD microbench + correctness gate for diffusion_sample_gpu (the per-denoise-step
// sampling kernel that the [dg-canvas-split] timer attributes ~1.6s to at C=2048).
//
// Isolated: no 26B model, no server. Builds standalone against diffusion_sampling.cu.
//   nvcc -O3 -arch=sm_86 -DDFLASH27B_BACKEND_CUDA -I server/src/diffusion \
//        server/test/bench_diffusion_sample.cu server/src/diffusion/diffusion_sampling.cu \
//        -o /tmp/bench_samp
//   /tmp/bench_samp <C> <target_ms>
//
// Hard gate (always): GPU argmax == CPU argmax (exact, committed output) and
//   entropy within rtol. This is the safety net for the coalescing refactor.
// Perf gate (RED→GREEN): mean kernel ms < target_ms. RED on the uncoalesced kernel.

#include "diffusion_sampling.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(2);} } while(0)

// Deterministic LCG in [0,1).
static inline float lcg(uint64_t & s){ s = s*6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((s>>40) & 0xFFFFFF) / (float)0x1000000; }

int main(int argc, char** argv){
    const int   C        = argc>1 ? atoi(argv[1]) : 2048;
    const float target_ms= argc>2 ? (float)atof(argv[2]) : 1e9f;
    const int   V        = 262144;          // gemma vocab
    const float temp_inv = 1.4f;
    const int   ITERS    = 30;

    std::printf("[bench] C=%d V=%d temp_inv=%.2f iters=%d target_ms=%.1f\n",
                C, V, temp_inv, ITERS, target_ms);

    // ── Host inputs: realistic-ish logits (random + a few sharp peaks per row) ──
    std::vector<float> h_logits((size_t)C*V);
    std::vector<float> h_u(C);
    uint64_t s = 0x1234567ULL;
    for (int p=0;p<C;++p){
        for (int v=0; v<V; ++v) h_logits[(size_t)p*V+v] = (lcg(s)-0.5f)*6.0f;  // ~U(-3,3)
        // inject a clear peak so argmax is unambiguous (no FP ties)
        int peak = (int)(lcg(s)*V); h_logits[(size_t)p*V+peak] = 30.0f + lcg(s);
        h_u[p] = lcg(s)*0.999999f + 1e-7f;
    }

    // ── CPU reference ──
    std::vector<int32_t> ref_argmax(C);
    std::vector<float>   ref_entropy(C);
    std::vector<int32_t> ref_sampled(C);
    for (int p=0;p<C;++p){
        const float* row=&h_logits[(size_t)p*V];
        float m=-1e38f; int am=0;
        for (int v=0;v<V;++v){ float z=row[v]*temp_inv; if(z>m){m=z;am=v;} }
        double Z=0.0, H=0.0;
        for (int v=0;v<V;++v){ double e=exp((double)(row[v]*temp_inv-m)); Z+=e; }
        double logZ=log(Z>0?Z:1e-38);
        for (int v=0;v<V;++v){ double e=exp((double)(row[v]*temp_inv-m)); double pr=e/Z;
            if(pr>0) H += -pr*log(pr); }
        double target=(double)h_u[p]*Z, cum=0.0; int sm=-1;
        for (int v=0;v<V;++v){ cum+=exp((double)(row[v]*temp_inv-m)); if(sm<0&&cum>=target){sm=v;break;} }
        ref_argmax[p]=am; ref_entropy[p]=(float)H; ref_sampled[p]=(sm<0?am:sm);
    }

    // ── Device ──
    float *d_logits,*d_u; int32_t *d_samp,*d_amax; float *d_ent;
    CK(cudaMalloc(&d_logits,(size_t)C*V*sizeof(float)));
    CK(cudaMalloc(&d_u,C*sizeof(float)));
    CK(cudaMalloc(&d_samp,C*sizeof(int32_t)));
    CK(cudaMalloc(&d_ent ,C*sizeof(float)));
    CK(cudaMalloc(&d_amax,C*sizeof(int32_t)));
    CK(cudaMemcpy(d_logits,h_logits.data(),(size_t)C*V*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_u,h_u.data(),C*sizeof(float),cudaMemcpyHostToDevice));

    using dflash::diffusion::diffusion_sample_gpu;
    // warmup
    diffusion_sample_gpu(d_logits,d_u,temp_inv,C,V,d_samp,d_ent,d_amax,0);
    CK(cudaDeviceSynchronize());

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i=0;i<ITERS;++i) diffusion_sample_gpu(d_logits,d_u,temp_inv,C,V,d_samp,d_ent,d_amax,0);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); ms/=ITERS;

    std::vector<int32_t> g_argmax(C),g_samp(C); std::vector<float> g_ent(C);
    CK(cudaMemcpy(g_argmax.data(),d_amax,C*sizeof(int32_t),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g_ent.data()  ,d_ent ,C*sizeof(float)  ,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(g_samp.data() ,d_samp,C*sizeof(int32_t),cudaMemcpyDeviceToHost));

    // ── Compare ──
    int argmax_bad=0, samp_bad=0; float ent_maxerr=0;
    for (int p=0;p<C;++p){
        if (g_argmax[p]!=ref_argmax[p]) argmax_bad++;
        if (g_samp[p]  !=ref_sampled[p]) samp_bad++;
        float e=std::fabs(g_ent[p]-ref_entropy[p]);
        float d=std::fabs(ref_entropy[p])+1e-6f;
        ent_maxerr=std::max(ent_maxerr, e/d);
    }
    std::printf("[bench] KERNEL_MS=%.2f  argmax_mismatch=%d/%d  entropy_rel_maxerr=%.4g  sampled_mismatch=%d/%d\n",
                ms, argmax_bad, C, ent_maxerr, samp_bad, C);

    // Hard correctness gate: committed output (argmax) must be exact; entropy within 1%.
    bool correct = (argmax_bad==0) && (ent_maxerr < 1e-2f);
    bool fast    = (ms < target_ms);
    std::printf("[bench] CORRECT=%s  PERF=%s  => %s\n",
                correct?"PASS":"FAIL", fast?"PASS":"FAIL",
                (correct&&fast)?"GREEN":"RED");
    return (correct&&fast)?0:1;
}
