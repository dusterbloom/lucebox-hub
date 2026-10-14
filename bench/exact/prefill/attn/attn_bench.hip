// Prefill attention microbench: one Qwen4exp full-attention layer (24 q heads x 256, 2 kv heads, QSA ratio 4,
// indexer budget 512 blocks), gfx1151 (lucebox4). Arms:
//   ours   : qsa3 rows + merge + attn from qsa.cu (exported launchers), queries 0..T-1 in one launch each
//   gufo   : WmmaCausalAttention<4,16,true> for queries >= 2048 (mask) + <16,16,false> for queries < 2048, the
//            split its executor makes (MIT)
// Same Q (F32 [T][24][256]), same K/V (F16), same per-query block selection (gufo bitmask, ours id rows as
// qsa_decode_ids writes them: 512 sorted blocks x 4 cells, future blocks -1, then the 3 tail cells).
// Selection: queries with <= 512 complete blocks see all of them; otherwise each query of a 4-query group picks
// 512 blocks from the group's random pool of P blocks (-pool, union size knob), or real ids from -ids <file>.
// Error: vs an F64 reference on -sample evenly spaced queries, all heads.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e)); exit(1);} } while (0)

constexpr int HQ = 24, HK = 2, D = 256, GQA = 12, R = 4, BUDGET = 512, NS = BUDGET * R + R - 1;

extern "C" int  ggml_cuda_qsa3_union_cap(int ns);
extern "C" void ggml_cuda_qsa3_union_launch(const int * ids, size_t i1, int n_q, int ns, int nk, int * srow, int * sflag,
        uint16_t * ublk, uint16_t * umask, int * ucount, int phases, int rows_variant, hipStream_t stream);
extern "C" void ggml_cuda_qsa3_attn_launch(const float * q, size_t q1, size_t q2, const uint16_t * pk, const uint16_t * pv,
        const uint16_t * mask, size_t m1, const uint16_t * ublk, const uint16_t * umask, const int * ucount, int n_q, int ns,
        int nk, int n_kv_head, float scale, float * out, size_t o1, size_t o2, hipStream_t stream);
extern "C" int  gufo_wmma_attention(const float * q, const uint16_t * k, const uint16_t * v, const uint32_t * mask,
        uint32_t mask_words, float * out, uint32_t n_tokens, uint32_t start_pos, hipStream_t st);

static uint16_t f2h(float f) { _Float16 h = (_Float16) f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float    h2f(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float) h; }

struct Set {
    std::vector<float>    q;            // [T][HQ][D]
    std::vector<uint16_t> kg, vg;       // gufo layout [T][HK][D]
    std::vector<uint16_t> kc, vc;       // ggml cache layout [HK][Tc][D]
    std::vector<uint16_t> pk, pv;       // packed (qsa_pack_keys / qsa_pack_values)
    std::vector<int32_t>  ids;          // [T][NS]
    std::vector<uint32_t> mask;         // [T][mask_words]
    std::vector<std::vector<int>> sel;  // per query, selected complete blocks (sorted)
    float * d_q; uint16_t * d_kg, * d_vg, * d_pk, * d_pv; int32_t * d_ids; uint32_t * d_mask;
};

static int T = 4096, Tc = 4096, mask_words = 0, g_rows = 2;

static void build_set(Set & s, std::mt19937 & rng, int pool, const std::vector<int32_t> * real_ids) {
    std::normal_distribution<float> nq(0.f, 0.5f), nv(0.f, 1.f);
    s.q.resize((size_t) T * HQ * D);
    for (auto & x : s.q) x = nq(rng);
    s.kg.resize((size_t) T * HK * D); s.vg.resize(s.kg.size());
    for (auto & x : s.kg) x = f2h(nq(rng));
    for (auto & x : s.vg) x = f2h(nv(rng));
    s.kc.assign((size_t) HK * Tc * D, 0); s.vc.assign(s.kc.size(), 0);
    for (int t = 0; t < T; ++t) for (int h = 0; h < HK; ++h) for (int d = 0; d < D; ++d) {
        s.kc[((size_t) h * Tc + t) * D + d] = s.kg[((size_t) t * HK + h) * D + d];
        s.vc[((size_t) h * Tc + t) * D + d] = s.vg[((size_t) t * HK + h) * D + d];
    }
    const int nblk = Tc / R;
    s.pk.resize((size_t) HK * nblk * 1024); s.pv.resize(s.pk.size());
    for (int h = 0; h < HK; ++h) for (int b = 0; b < nblk; ++b) {
        uint16_t * bk = s.pk.data() + ((size_t) h * nblk + b) * 1024, * bv = s.pv.data() + ((size_t) h * nblk + b) * 1024;
        for (int j = 0; j < R; ++j) for (int d = 0; d < D; ++d) {
            const uint16_t kk = s.kc[((size_t) h * Tc + 4*b + j) * D + d], vv = s.vc[((size_t) h * Tc + 4*b + j) * D + d];
            bk[(d / 16) * 64 + j * 16 + (d % 16)] = kk;
            bv[d * 4 + j] = vv;
        }
    }
    // selection
    s.sel.assign(T, {});
    s.ids.assign((size_t) T * NS, -1);
    s.mask.assign((size_t) T * mask_words, 0);
    const int nb_total = T / R;
    for (int g = 0; g < (T + 3) / 4; ++g) {
        const int nvis0 = (4*g + 1) / R;   // complete blocks visible to the group's first query
        std::vector<int> poolv;            // the group's shared candidate pool (union-size knob)
        if (nvis0 > BUDGET) {
            std::vector<int> all(nvis0); for (int i = 0; i < nvis0; ++i) all[i] = i;
            std::shuffle(all.begin(), all.end(), rng);
            poolv.assign(all.begin(), all.begin() + std::min(pool, nvis0));
        } else {
            for (int b = 0; b < nvis0; ++b) poolv.push_back(b);
        }
        for (int qi = 0; qi < 4 && 4*g + qi < T; ++qi) {
            const int t = 4*g + qi, nvis = (t + 1) / R;
            std::vector<int> & sel = s.sel[t];
            int32_t * row = s.ids.data() + (size_t) t * NS;
            if (real_ids) {   // verbatim row; sel = its complete selected blocks (the tail cells are implicit for gufo)
                memcpy(row, real_ids->data() + (size_t) t * NS, (size_t) NS * 4);
                std::set<int> blocks;
                for (int j = 0; j < BUDGET * R; ++j) if (row[j] >= 0 && row[j] / R * R + R - 1 <= t) blocks.insert(row[j] / R);
                sel.assign(blocks.begin(), blocks.end());
            } else {
                if (nvis <= BUDGET) {
                    for (int b = 0; b < nvis; ++b) sel.push_back(b);
                } else {
                    std::vector<int> p = poolv;
                    for (int b = nvis0; b < nvis; ++b) p.push_back(b);   // this query's extra complete block(s)
                    std::shuffle(p.begin(), p.end(), rng);
                    sel.assign(p.begin(), p.begin() + BUDGET);
                    std::sort(sel.begin(), sel.end());
                }
                // ids row (qsa_decode_ids semantics): 512 sorted blocks, future blocks fill the rest with -1 cells
                std::vector<int> blocks = sel;
                for (int b = nvis; (int) blocks.size() < BUDGET && b < nb_total; ++b) blocks.push_back(b);
                while ((int) blocks.size() < BUDGET) blocks.push_back(nb_total);   // never visible
                std::sort(blocks.begin(), blocks.end());
                for (int j = 0; j < BUDGET * R; ++j) {
                    const int first = R * blocks[j / R];
                    row[j] = (first + R - 1 <= t) ? first + j % R : -1;
                }
                const int br = ((t + 1) / R) * R;
                for (int j = 0; j < R - 1; ++j) row[BUDGET * R + j] = (br + j <= t) ? br + j : -1;
            }
            for (int b : sel) s.mask[(size_t) t * mask_words + b / 32] |= 1u << (b % 32);
        }
    }
    CHECK(hipMalloc(&s.d_q, s.q.size() * 4));      CHECK(hipMemcpy(s.d_q, s.q.data(), s.q.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&s.d_kg, s.kg.size() * 2));    CHECK(hipMemcpy(s.d_kg, s.kg.data(), s.kg.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&s.d_vg, s.vg.size() * 2));    CHECK(hipMemcpy(s.d_vg, s.vg.data(), s.vg.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&s.d_pk, s.pk.size() * 2));    CHECK(hipMemcpy(s.d_pk, s.pk.data(), s.pk.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&s.d_pv, s.pv.size() * 2));    CHECK(hipMemcpy(s.d_pv, s.pv.data(), s.pv.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&s.d_ids, s.ids.size() * 4));  CHECK(hipMemcpy(s.d_ids, s.ids.data(), s.ids.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&s.d_mask, s.mask.size() * 4)); CHECK(hipMemcpy(s.d_mask, s.mask.data(), s.mask.size() * 4, hipMemcpyHostToDevice));
}

// F64 reference for query t, all heads: softmax over the valid cells of its ids row.
static void reference(const Set & s, int t, std::vector<double> & out) {
    const int32_t * row = s.ids.data() + (size_t) t * NS;
    std::vector<int> cells;
    for (int j = 0; j < NS; ++j) if (row[j] >= 0) cells.push_back(row[j]);
    std::sort(cells.begin(), cells.end()); cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    out.assign((size_t) HQ * D, 0.0);
    std::vector<double> sc(cells.size());
    for (int h = 0; h < HQ; ++h) {
        const int kvh = h / GQA;
        const float * q = s.q.data() + ((size_t) t * HQ + h) * D;
        double m = -1e300;
        for (size_t i = 0; i < cells.size(); ++i) {
            const uint16_t * k = s.kg.data() + ((size_t) cells[i] * HK + kvh) * D;
            double acc = 0; for (int d = 0; d < D; ++d) acc += (double) q[d] * (double) h2f(k[d]);
            sc[i] = acc / 16.0; m = std::max(m, sc[i]);
        }
        double l = 0; double * o = out.data() + (size_t) h * D;
        for (size_t i = 0; i < cells.size(); ++i) {
            const double p = std::exp(sc[i] - m); l += p;
            const uint16_t * v = s.vg.data() + ((size_t) cells[i] * HK + kvh) * D;
            for (int d = 0; d < D; ++d) o[d] += p * (double) h2f(v[d]);
        }
        for (int d = 0; d < D; ++d) o[d] /= l;
    }
}

struct Scratch { int * srow, * sflag, * ucount; uint16_t * ublk, * umask; float * out; };

int main(int argc, char ** argv) {
    int reps = 20, nsets = 3, nsample = 96, pool = 640; const char * ids_file = nullptr; const char * dump = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-T")) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-L")) nsets = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-sample")) nsample = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-pool")) pool = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ids")) ids_file = argv[++i];
        else if (!strcmp(argv[i], "-dump")) dump = argv[++i];
        else if (!strcmp(argv[i], "-rows")) g_rows = atoi(argv[++i]);
    }
    Tc = (T + 3) / 4 * 4;
    mask_words = ((T / R + 31) / 32 * 32) / 32;
    std::vector<int32_t> real_ids;
    if (ids_file) {
        real_ids.resize((size_t) T * NS);
        FILE * f = fopen(ids_file, "rb"); if (!f || fread(real_ids.data(), 4, real_ids.size(), f) != real_ids.size()) { fprintf(stderr, "bad ids file\n"); return 1; }
        fclose(f);
    }
    hipStream_t st; CHECK(hipStreamCreate(&st));
    std::mt19937 rng(7);
    std::vector<Set> sets(nsets);
    for (int i = 0; i < nsets; ++i) build_set(sets[i], rng, pool, ids_file ? &real_ids : nullptr);
    // union size per 4-query group (blocks incl. tail), what both kernels sweep
    {
        double tot = 0, tot_sparse = 0; int n_sparse = 0;
        for (int g = 0; g < T / 4; ++g) {
            std::set<int> u;
            for (int qi = 0; qi < 4; ++qi) { const int t = 4*g+qi; for (int b : sets[0].sel[t]) u.insert(b); u.insert(t / R); }
            tot += u.size(); if (4*g >= 2048) { tot_sparse += u.size(); ++n_sparse; }
        }
        printf("T=%d sets=%d pool=%d%s: mean union blocks/group %.1f (sparse groups %.1f, n=%d) -> mean 16-key chunks %.1f\n",
               T, nsets, pool, ids_file ? " (real ids)" : "", tot / (T/4), n_sparse ? tot_sparse / n_sparse : 0.0, n_sparse, tot / (T/4) / 4);
    }
    const int ngroups = (T + 3) / 4, cap = ggml_cuda_qsa3_union_cap(NS);
    Scratch sc;
    CHECK(hipMalloc(&sc.srow, (size_t) T * NS * 4)); CHECK(hipMalloc(&sc.sflag, (size_t) T * 4)); CHECK(hipMalloc(&sc.ucount, (size_t) ngroups * 4));
    CHECK(hipMalloc(&sc.ublk, (size_t) ngroups * cap * 2)); CHECK(hipMalloc(&sc.umask, (size_t) ngroups * cap * 2));
    CHECK(hipMalloc(&sc.out, (size_t) T * HQ * D * 4));
    const size_t q1 = (size_t) HQ * D * 4, q2 = (size_t) D * 4;

    auto run_ours = [&](const Set & s, hipEvent_t * ev) {
        CHECK(hipEventRecord(ev[0], st));
        ggml_cuda_qsa3_union_launch(s.d_ids, (size_t) NS * 4, T, NS, Tc, sc.srow, sc.sflag, sc.ublk, sc.umask, sc.ucount, 1, g_rows, st);
        CHECK(hipEventRecord(ev[1], st));
        ggml_cuda_qsa3_union_launch(s.d_ids, (size_t) NS * 4, T, NS, Tc, sc.srow, sc.sflag, sc.ublk, sc.umask, sc.ucount, 2, g_rows, st);
        CHECK(hipEventRecord(ev[2], st));
        ggml_cuda_qsa3_attn_launch(s.d_q, q1, q2, s.d_pk, s.d_pv, nullptr, 0, sc.ublk, sc.umask, sc.ucount, T, NS, Tc, HK, 1.0f / 16.0f,
                                   sc.out, q2, q1, st);
        CHECK(hipEventRecord(ev[3], st));
    };
    auto run_gufo = [&](const Set & s, hipEvent_t * ev) {
        const int dense = std::min(T, 2048);
        CHECK(hipEventRecord(ev[0], st));
        if (T > dense) {
            if (!gufo_wmma_attention(s.d_q + (size_t) dense * HQ * D, s.d_kg, s.d_vg, s.d_mask + (size_t) dense * mask_words, mask_words,
                                     sc.out + (size_t) dense * HQ * D, T - dense, dense, st)) { fprintf(stderr, "gufo sparse refused\n"); exit(1); }
        }
        CHECK(hipEventRecord(ev[1], st));
        if (!gufo_wmma_attention(s.d_q, s.d_kg, s.d_vg, nullptr, mask_words, sc.out, dense, 0, st)) { fprintf(stderr, "gufo dense refused\n"); exit(1); }
        CHECK(hipEventRecord(ev[2], st));
        CHECK(hipEventRecord(ev[3], st));
    };
    const char * ours_ph[3] = {"rows", "merge", "attn"}, * gufo_ph[3] = {"sparse<4,16>", "dense<16,16>", ""};

    for (int vv = 1; vv < 3; ++vv) { // union builder parity: legacy rows (0) vs compacting rows (1) / bitmap (2): byte-identical ublk/umask/ucount
        std::vector<uint16_t> b0((size_t) ngroups * cap), m0(b0.size()), b1(b0.size()), m1(b0.size());
        std::vector<int> c0(ngroups), c1(ngroups);
        for (int v : { 0, vv }) {
            CHECK(hipMemset(sc.ublk, 0xEE, b0.size() * 2)); CHECK(hipMemset(sc.umask, 0xEE, b0.size() * 2)); CHECK(hipMemset(sc.ucount, 0xEE, ngroups * 4));
            ggml_cuda_qsa3_union_launch(sets[0].d_ids, (size_t) NS * 4, T, NS, Tc, sc.srow, sc.sflag, sc.ublk, sc.umask, sc.ucount, 3, v, st);
            CHECK(hipStreamSynchronize(st));
            CHECK(hipMemcpy((v ? b1 : b0).data(), sc.ublk, b0.size() * 2, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy((v ? m1 : m0).data(), sc.umask, b0.size() * 2, hipMemcpyDeviceToHost));
            CHECK(hipMemcpy((v ? c1 : c0).data(), sc.ucount, ngroups * 4, hipMemcpyDeviceToHost));
        }
        size_t bad = 0; long long tot = 0;
        for (int g = 0; g < ngroups; ++g) {
            tot += c0[g];
            if (c0[g] != c1[g]) { ++bad; continue; }
            for (int j = 0; j < c0[g]; ++j) if (b0[(size_t) g * cap + j] != b1[(size_t) g * cap + j] || m0[(size_t) g * cap + j] != m1[(size_t) g * cap + j]) { ++bad; break; }
        }
        printf("union legacy vs variant %d: %zu of %d groups differ (mean padded union %.1f blocks)\n", vv, bad, ngroups, (double) tot / ngroups);
    }
    // correctness vs F64 reference
    std::vector<int> sample(nsample);
    for (int i = 0; i < nsample; ++i) sample[i] = (int) (((double) i + 0.5) * T / nsample);
    std::vector<std::vector<double>> ref(nsample);
    for (int i = 0; i < nsample; ++i) reference(sets[0], sample[i], ref[i]);
    std::vector<float> got((size_t) T * HQ * D);
    hipEvent_t ev[4]; for (auto & evt : ev) CHECK(hipEventCreate(&evt));
    std::vector<float> out_ours, out_gufo;
    for (int arm = 0; arm < 2; ++arm) {
        CHECK(hipMemset(sc.out, 0xFF, (size_t) T * HQ * D * 4));
        if (arm == 0) run_ours(sets[0], ev); else run_gufo(sets[0], ev);
        CHECK(hipStreamSynchronize(st));
        CHECK(hipMemcpy(got.data(), sc.out, got.size() * 4, hipMemcpyDeviceToHost));
        (arm == 0 ? out_ours : out_gufo) = got;
        double max_abs = 0, se = 0, ref_rms = 0; size_t n = 0, nan = 0; int worst_t = -1;
        for (int i = 0; i < nsample; ++i) {
            const float * o = got.data() + (size_t) sample[i] * HQ * D;
            for (size_t j = 0; j < (size_t) HQ * D; ++j) {
                if (!std::isfinite(o[j])) { ++nan; continue; }
                const double e = std::fabs((double) o[j] - ref[i][j]);
                if (e > max_abs) { max_abs = e; worst_t = sample[i]; }
                se += e * e; ref_rms += ref[i][j] * ref[i][j]; ++n;
            }
        }
        if (dump) { char fn[512]; snprintf(fn, sizeof fn, "%s.%s.bin", dump, arm == 0 ? "ours" : "gufo"); FILE * f = fopen(fn, "wb"); fwrite(got.data(), 4, got.size(), f); fclose(f); }
        printf("%-6s vs F64 ref (%d queries x 24 heads): max_abs %.3e (t=%d) rms_err %.3e ref_rms %.3e nonfinite %zu\n",
               arm == 0 ? "ours" : "gufo", nsample, max_abs, worst_t, std::sqrt(se / n), std::sqrt(ref_rms / n), nan);
    }
    { // ours vs gufo
        double max_abs = 0; size_t n_exact = 0, worst = 0;
        for (size_t j = 0; j < out_ours.size(); ++j) {
            const double e = std::fabs(out_ours[j] - out_gufo[j]);
            if (e > max_abs) { max_abs = e; worst = j; }
            n_exact += out_ours[j] == out_gufo[j];
        }
        const int wt = (int) (worst / ((size_t) HQ * D)), wh = (int) (worst / D % HQ);
        std::vector<double> r; reference(sets[0], wt, r);
        double eo = 0, eg = 0;
        for (size_t j = 0; j < (size_t) HQ * D; ++j) {
            eo = std::max(eo, std::fabs(out_ours[(size_t) wt * HQ * D + j] - r[j]));
            eg = std::max(eg, std::fabs(out_gufo[(size_t) wt * HQ * D + j] - r[j]));
        }
        printf("ours vs gufo: max_abs %.3e at t=%d h=%d (there: ours max_abs vs ref %.3e, gufo %.3e), bit-exact %.2f%%\n",
               max_abs, wt, wh, eo, eg, 100.0 * n_exact / out_ours.size());
    }

    // timing
    for (int arm = 0; arm < 2; ++arm) {
        std::vector<std::vector<float>> ph(3);
        for (int r = -2; r < reps; ++r) {
            const Set & s = sets[(r + 2) % nsets];
            if (arm == 0) run_ours(s, ev); else run_gufo(s, ev);
            CHECK(hipStreamSynchronize(st));
            if (r < 0) continue;
            for (int p = 0; p < 3; ++p) { float ms; CHECK(hipEventElapsedTime(&ms, ev[p], ev[p+1])); ph[p].push_back(ms); }
        }
        auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; };
        auto mn  = [](std::vector<float> v) { return *std::min_element(v.begin(), v.end()); };
        std::vector<float> tot(reps, 0.f); for (int p = 0; p < 3; ++p) for (int r = 0; r < reps; ++r) tot[r] += ph[p][r];
        printf("%-6s total median %.3f ms (min %.3f) over %d reps:", arm == 0 ? "ours" : "gufo", med(tot), mn(tot), reps);
        const char ** names = arm == 0 ? ours_ph : gufo_ph;
        for (int p = 0; p < (arm == 0 ? 3 : 2); ++p) printf("  %s %.3f (min %.3f)", names[p], med(ph[p]), mn(ph[p]));
        printf("\n");
    }
    return 0;
}
