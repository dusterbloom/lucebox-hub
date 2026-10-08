#include "qwen4exp_internal.h"
#include "qwen4exp_graph.h"
#include "qwen4exp_cache.h"
#include "qwen4exp_pipeline.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
extern "C" uint64_t ggml_backend_cuda_qwen_graph_probe_count(int kind);
extern "C" size_t ggml_backend_cuda_get_expert_row_warps_launch_count(void);
extern "C" size_t ggml_backend_cuda_get_qwen_shared_overlap_fork_count(void);
extern "C" size_t ggml_backend_cuda_get_qwen_shared_overlap_join_count(void);
extern "C" int ggml_backend_cuda_set_hc_lo_q8_for_test(int mode);
extern "C" size_t ggml_backend_cuda_get_hc_lo_q8_launch_count(void);
extern "C" uint64_t ggml_backend_cuda_get_hc_lo_q8_replay_sites(void);
extern "C" int ggml_backend_cuda_set_gdn_ab_exact_for_test(int mode);
extern "C" uint64_t ggml_backend_cuda_get_gdn_ab_host_layers(void);
extern "C" uint64_t ggml_backend_cuda_get_gdn_ab_replay_layers(void);
extern "C" int ggml_backend_cuda_set_producer_q8_for_test(int mode);
extern "C" uint64_t ggml_backend_cuda_get_producer_q8_sites(int kind);
extern "C" size_t ggml_backend_cuda_get_hc_down_inject_launch_count(void);
extern "C" uint64_t ggml_backend_cuda_get_hc_down_inject_replay_pairs(void);
extern "C" int ggml_backend_cuda_set_hc_upmix_row8_for_test(int mode);
extern "C" size_t ggml_backend_cuda_get_hc_upmix_row8_launch_count(void);
extern "C" uint64_t ggml_backend_cuda_get_hc_upmix_row8_replay_sites(void);
extern "C" int ggml_backend_cuda_set_shared_epilogue_for_test(int);
extern "C" uint64_t ggml_backend_cuda_get_shared_epilogue_sites(int);
using namespace luce::common;
static std::vector<int32_t> ids(const char * path) {
    std::ifstream f(path); std::vector<int32_t> v; int t;
    while (f >> t) v.push_back(t);
    if (!f.eof() || v.empty()) throw std::runtime_error("invalid tokens");
    return v;
}
static int top(const std::vector<float> & v) {
    return std::max_element(v.begin(), v.end()) - v.begin();
}
int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    const auto prompt = ids(argv[2]); const int n = std::stoi(argv[3]), reps = std::stoi(argv[4]);
    const auto enabled=[](const char * k,const char * v){const char * e=getenv(k);return e&&std::string(e)==v;};
    // LUCE_QWEN_PIPELINE=1 requires LUCE_QWEN_SHARED_OVERLAP=0 (forward_impl's pipe gate excludes shared overlap);
    // the baseline recipe otherwise always pins SHARED_OVERLAP=1.
    const bool pipeline_requested = enabled("LUCE_QWEN_PIPELINE","1");
    const bool shared_overlap_env_ok = enabled("LUCE_QWEN_SHARED_OVERLAP","0") || enabled("LUCE_QWEN_SHARED_OVERLAP","1");
    if (!enabled("LUCE_QWEN_GRAPH_SH","1") || !enabled("LUCE_QWEN_HC_DOWN_INJECT","1") ||
        !enabled("LUCE_QWEN_EXPERT_ROW_WARPS","8") ||
        !shared_overlap_env_ok || (pipeline_requested && enabled("LUCE_QWEN_SHARED_OVERLAP","1")) ||
        !enabled("LUCE_QWEN_QSA_CONT_ELISION","1") || !enabled("LUCE_QWEN_EXACT_ROUTER_SUFFIX","1") ||
        !enabled("LUCE_HOST_PREFILL_GUARDS","1") || !enabled("LUCE_QWEN_HC_SCALE_SILU","1") ||
        !enabled("MEASURE_GPU_ARGMAX","1") || !enabled("LUCE_QWEN_GDN_AB_EXACT","1") ||
        !enabled("LUCE_QWEN_HC_LO_Q8","1") || !enabled("LUCE_QWEN_PRODUCER_Q8","1") ||
        !enabled("LUCE_QWEN_HC_UPMIX_ROW8","1") || !enabled("LUCE_QWEN_SHARED_EPILOGUE","1")) return 2;
    const bool golden = n == 128 && reps == 2, native = n == 256 && reps == 8;
    const bool census=n==64 && reps==1;
    if (!golden && !native && !census) return 2;
    if(census && (!getenv("LUCE_SHARED_EPILOGUE_CENSUS_MODE") || (getenv("MEASURE_LOGITS") || getenv("MEASURE_STATE"))))return 2;
    if (native && (getenv("MEASURE_LOGITS") || getenv("MEASURE_STATE"))) return 2;
    if (golden && (!getenv("MEASURE_LOGITS") || !getenv("MEASURE_STATE"))) return 2;
    std::vector<int32_t> follow; if (const char * p=getenv("MEASURE_FOLLOW")) follow=ids(p);
    if (prompt.size()!=7235 || follow.size()<(size_t)n || prompt.size()+n>8192) return 2;
    auto counts=[] { return std::vector<uint64_t>{
        ggml_backend_cuda_qwen_graph_probe_count(0), ggml_backend_cuda_qwen_graph_probe_count(1),
        ggml_backend_cuda_qwen_graph_probe_count(2), ggml_backend_cuda_qwen_graph_probe_count(3),
        ggml_backend_cuda_get_expert_row_warps_launch_count(),
        ggml_backend_cuda_get_qwen_shared_overlap_fork_count(),
        ggml_backend_cuda_get_qwen_shared_overlap_join_count(),
        ggml_backend_cuda_get_hc_down_inject_launch_count(),
        ggml_backend_cuda_get_hc_down_inject_replay_pairs(),
        ggml_backend_cuda_get_hc_lo_q8_launch_count(),
        ggml_backend_cuda_get_hc_lo_q8_replay_sites(),
        ggml_backend_cuda_get_gdn_ab_host_layers(),
        ggml_backend_cuda_get_gdn_ab_replay_layers(),
        ggml_backend_cuda_get_producer_q8_sites(0), ggml_backend_cuda_get_producer_q8_sites(1),
        ggml_backend_cuda_get_producer_q8_sites(2), ggml_backend_cuda_get_producer_q8_sites(3),
        ggml_backend_cuda_get_hc_upmix_row8_launch_count(),
        ggml_backend_cuda_get_hc_upmix_row8_replay_sites(),
        ggml_backend_cuda_get_shared_epilogue_sites(0), ggml_backend_cuda_get_shared_epilogue_sites(1),
        ggml_backend_cuda_get_shared_epilogue_sites(2)}; };
    auto backend=ggml_backend_cuda_init(0); Qwen4ExpWeights w;
    if (!backend || !load_qwen4exp_gguf(argv[1],backend,w,"0",false,QWEN4EXP_MTP_VOCAB)) return 1;
    Qwen4ExpCache cache;
    if (!create_qwen4exp_cache(backend,w,8192,GGML_TYPE_F16,cache,false,1,false)) return 1;
    std::vector<float> logits; std::vector<int> picks;
    auto now=[] {return std::chrono::steady_clock::now();};
    const int schedule[8]={0,1,0,1,1,0,1,0};
    const int runs=census?1:(native?10:2); auto previous=counts();
    for (int run=0; run<runs; ++run) {
        const bool warm=native && run<2; const int rep=warm?run:run-(native?2:0);
        const int mode=census?std::atoi(getenv("LUCE_SHARED_EPILOGUE_CENSUS_MODE")):(warm?run:(native?schedule[rep]:rep));
        if(mode<0 || mode>1)return 2;
        ggml_backend_cuda_set_graphs_disabled_override(true);
        reset_qwen4exp_state(backend,cache);
        ggml_backend_cuda_set_hc_lo_q8_for_test(1);
        ggml_backend_cuda_set_producer_q8_for_test(1);
        ggml_backend_cuda_set_hc_upmix_row8_for_test(1);
        ggml_backend_cuda_set_shared_epilogue_for_test(mode);
        ggml_backend_cuda_set_gdn_ab_exact_for_test(1);
        for(int pos=0;pos<(int)prompt.size()-1;pos+=4096) {
            int count=std::min(4096,(int)prompt.size()-1-pos);
            if(!qwen4exp_forward(backend,w,cache,prompt.data()+pos,count,pos,logits).ok)return 1;
        }
        ggml_backend_cuda_set_graphs_disabled_override(false);
        const auto before=counts();
        for(size_t i=0;i<before.size();++i) if(i!=4 && i!=9 && i!=14 && before[i]!=previous[i]) return 3;
        if(before[4]-previous[4]!=4) return 3;
        const uint64_t prefill_lo=before[9]-previous[9];
        const uint64_t prefill_producer_hc=before[14]-previous[14];
        if (prefill_lo != 4 || prefill_producer_hc != 4) return 3;
        std::FILE * dump=nullptr;
        if(const char * p=getenv("MEASURE_LOGITS")) {
            std::string target=std::string(p)+".mode"+std::to_string(mode);
            dump=std::fopen(target.c_str(),"wb"); if(!dump)return 1;
        }
        std::printf("[schedule] kind=%s rep=%d mode=%d prefill_lo=%llu prefill_producer_hc=%llu\n",warm?"warm":"measure",rep,mode,(unsigned long long)prefill_lo,(unsigned long long)prefill_producer_hc);
        std::fflush(stdout);
        picks.clear(); int32_t token=prompt.back(); auto start=now();
        const bool pipeline_on = enabled("LUCE_QWEN_PIPELINE","1");
        if (pipeline_on && !dump) {
            // LUCE_QWEN_PIPELINE=1: forced-follow greedy session, lookahead=1 (see qwen4exp_pipeline.h).
            Qwen4ExpPipeline * pipe = qwen4exp_pipeline_create(backend, w, cache);
            if (!pipe) { std::fprintf(stderr, "[pipe-dbg] create failed\n"); return 1; }
            if (!qwen4exp_pipeline_begin(*pipe, token)) { std::fprintf(stderr, "[pipe-dbg] begin failed\n"); qwen4exp_pipeline_destroy(pipe); return 1; }
            for (int i = 0; i < n; ++i) {
                if (!qwen4exp_pipeline_step(*pipe, &follow[i])) { std::fprintf(stderr, "[pipe-dbg] step failed i=%d\n", i); qwen4exp_pipeline_destroy(pipe); return 1; }
                int32_t pick = -1;
                if (!qwen4exp_pipeline_wait(*pipe, i, pick)) { std::fprintf(stderr, "[pipe-dbg] wait failed i=%d\n", i); qwen4exp_pipeline_destroy(pipe); return 1; }
                if (pick < 0) { std::fprintf(stderr, "[pipe-dbg] pick<0 i=%d\n", i); qwen4exp_pipeline_destroy(pipe); return 1; }
                picks.push_back(pick);
            }
            if (!qwen4exp_pipeline_end(*pipe)) { std::fprintf(stderr, "[pipe-dbg] end failed\n"); qwen4exp_pipeline_destroy(pipe); return 1; }
            qwen4exp_pipeline_destroy(pipe);
        } else {
            for(int i=0;i<n;++i) {
                int32_t gpu=-1;
                if(!qwen4exp_forward(backend,w,cache,&token,1,(int)prompt.size()-1+i,logits,nullptr,false,false,false,false,nullptr,&gpu).ok)return 1;
                if(logits.empty()&&dump){logits.resize(w.n_vocab);ggml_backend_tensor_get(cache.decode_workspace.logits,logits.data(),0,ggml_nbytes(cache.decode_workspace.logits));}
                for(float x:logits)if(!std::isfinite(x))return 1;
                if(gpu<0||(!logits.empty()&&gpu!=top(logits)))return 1;
                picks.push_back(gpu); if(dump&&std::fwrite(logits.data(),sizeof(float),logits.size(),dump)!=logits.size())return 1;
                token=follow[i];
            }
        }
        if(const char * p=getenv("MEASURE_STATE")) {
            std::string target=std::string(p)+".mode"+std::to_string(mode); std::FILE * f=std::fopen(target.c_str(),"wb");if(!f)return 1;
            for(const auto * tensors:{&cache.ssm_state,&cache.conv_state,&cache.ple_conv_state})for(auto * tensor:*tensors){
                std::vector<unsigned char>b(ggml_nbytes(tensor));ggml_backend_tensor_get(tensor,b.data(),0,b.size());
                if(std::fwrite(b.data(),1,b.size(),f)!=b.size())return 1;}
            if(std::fclose(f))return 1;
        }
        const double ms=std::chrono::duration<double,std::milli>(now()-start).count();
        const auto after=counts(); std::vector<uint64_t>d(after.size());
        for(size_t i=0;i<d.size();++i)d[i]=after[i]-before[i];
        const uint64_t host=d[0]+d[1], logical_sh=d[5]+48*d[2], logical_hc=d[7]+d[8],
            logical_lo=d[9]+d[10], logical_ab=d[11]+d[12],
            logical_pg=d[13]+d[15], logical_ph=d[14]+d[16];
        // Pipelining splits each token's graph into two ggml_backend_graph_compute_async calls (pre/post PLE),
        // which does not match the single-call-per-token launch-count formulas below (those validate OTHER
        // fusions' host/replay bookkeeping, orthogonal to pipelining correctness). Pipelining's own gate is
        // bit-identical [measure_tokens] vs the non-pipelined reference, checked by the caller.
        const bool shared_overlap_on = enabled("LUCE_QWEN_SHARED_OVERLAP","1");
        if (!pipeline_requested && shared_overlap_on && (d[0]+d[1]+d[2]!=(uint64_t)n || !d[1] || !d[2] || d[0]!=d[3] || d[1]!=d[3] ||
           d[4]!=96*host || d[5]!=48*host || d[6]!=d[5] || logical_sh!=48*(uint64_t)n ||
           d[7]!=95*host || d[8]!=95*d[2] || logical_hc!=95*(uint64_t)n ||
           d[9]!=97*host || d[10]!=97*d[2] || logical_lo!=97*(uint64_t)n ||
           d[11]!=36*host || d[12]!=36*d[2] || logical_ab!=36*(uint64_t)n ||
           d[13]!=36*host || d[15]!=36*d[2] || logical_pg!=36*(uint64_t)n ||
           d[14]!=95*host || d[16]!=95*d[2] || logical_ph!=95*(uint64_t)n ||
           d[17]!=96*host || d[18]!=96*d[2] || d[17]+d[18]!=96*(uint64_t)n ||
           d[19]!=(mode?48*host:0) || d[20]!=(mode?48*d[2]:0) || d[21]!=d[19]))return 3;
        std::printf("[shared_epilogue_counts] rep=%d mode=%d host=%llu replay=%llu logical=%llu skipped_pairs_host=%llu\n",rep,mode,(unsigned long long)d[19],(unsigned long long)d[20],(unsigned long long)(d[19]+d[20]),(unsigned long long)d[21]);
        std::printf("[upmix_row8_counts] rep=%d mode=%d host=%llu replay=%llu logical=%llu\n", rep,mode,
            (unsigned long long)d[17],(unsigned long long)d[18],(unsigned long long)(d[17]+d[18]));
        previous=after; if(dump&&std::fclose(dump))return 1;
        std::printf("[graph_producer_q8_counts] rep=%d mode=%d eager=%llu capture=%llu replay=%llu seals=%llu expert_host=%llu sh_host=%llu sh_logical=%llu hc_host=%llu hc_replay=%llu hc_logical=%llu lo_host=%llu lo_replay=%llu lo_logical=%llu ab_host=%llu ab_replay=%llu ab_logical=%llu producer_gdn_host=%llu producer_gdn_replay=%llu producer_gdn_logical=%llu producer_hc_host=%llu producer_hc_replay=%llu producer_hc_logical=%llu\n",
            rep,mode,(unsigned long long)d[0],(unsigned long long)d[1],(unsigned long long)d[2],(unsigned long long)d[3],
            (unsigned long long)d[4],(unsigned long long)d[5],(unsigned long long)logical_sh,
            (unsigned long long)d[7],(unsigned long long)d[8],(unsigned long long)logical_hc,
            (unsigned long long)d[9],(unsigned long long)d[10],(unsigned long long)logical_lo,
            (unsigned long long)d[11],(unsigned long long)d[12],(unsigned long long)logical_ab,
            (unsigned long long)d[13],(unsigned long long)d[15],(unsigned long long)logical_pg,
            (unsigned long long)d[14],(unsigned long long)d[16],(unsigned long long)logical_ph);
        std::printf("[%s] rep=%d mode=%d tokens=%d decode_ms=%.6f ms_token=%.6f\n",warm?"warm":"measure",rep,mode,n,ms,ms/n);
        std::printf("[%s_tokens] rep=%d mode=%d",warm?"warm":"measure",rep,mode);for(int t:picks)std::printf(" %d",t);std::puts("");std::fflush(stdout);
    }
    free_qwen4exp_cache(cache);free_qwen4exp_weights(w);ggml_backend_free(backend);return 0;
}
