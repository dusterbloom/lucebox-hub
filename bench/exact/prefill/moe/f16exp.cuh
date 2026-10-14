// Included at the end of moe_bench.hip's main (-DMOE_EXP='"f16exp.cuh"'): the F16 single-launch kernel arm, a
// double-precision reference of the whole layer on the GPU, the error table of every arm against it, and the
// bf16 -> F16 activation conversion the F16 kernel needs once per layer.
{
    // ---- arm: mmb_routed_glu_f16_kernel (F16 activations D.xh = the same values as the bf16 D.xb) ----
    const int n_max = n_rows / BN + E + 1;
    uint32_t * tiles; int * ctl; CHECK(hipMalloc(&tiles, n_max * 4)); CHECK(hipMalloc(&ctl, 8));
    std::vector<float> new_f32((size_t) n_rows * M); std::vector<uint16_t> new_bf16((size_t) n_rows * M);
    auto runf = [&](int grid, bool f32) {
        mmb_build_tiles<<<1, 1024, 0, st>>>(D.bounds, tiles, ctl, E, n_max, BN);
        mmb_routed_glu_f16_kernel<BN, WT><<<grid, MMB_NT, 0, st>>>(D.wg, D.wu, expert_bytes, (const _Float16 *) D.xh, f32 ? D.d : nullptr, D.dh, f32, D.ids_src, D.ids_dst, D.bounds, tiles, ctl, M, K);
    };
    CHECK(hipMemset(D.d, 0, (size_t) n_rows * M * 4)); CHECK(hipMemset(D.dh, 0, (size_t) n_rows * M * 2));
    double fms = time_ms([&] { runf(80, true); }, st); grab(new_f32, new_bf16); report("ours F16 single-launch (F32+bf16 out) grid 80", fms, &new_f32, &new_bf16);
    for (int grid : {60, 80, 120, 160}) {
        CHECK(hipMemset(D.dh, 0, (size_t) n_rows * M * 2));
        char name[64]; snprintf(name, sizeof name, "ours F16 single-launch bf16 out, grid %d", grid);
        fms = time_ms([&] { runf(grid, false); }, st); grab(out_f32, out_bf16); report(name, fms, nullptr, &out_bf16);
    }
    if (memcmp(out_bf16.data(), new_bf16.data(), out_bf16.size() * 2) != 0) printf("| %-44s | | | | WARNING: bf16-only run differs from the F32+bf16 run |\n", "ours F16 in-situ check");

    // ---- bf16 -> F16 activation conversion (once per layer in the server) ----
    {
        _Float16 * xc; CHECK(hipMalloc(&xc, (size_t) T * K * 2));
        const size_t n = (size_t) T * K;
        double cms = time_ms([&] { cvt_bf16_f16_kernel<<<(unsigned) ((n / 8 + 255) / 256), 256, 0, st>>>(D.xb, xc, n); }, st);
        std::vector<uint16_t> xcv(n); CHECK(hipMemcpy(xcv.data(), xc, n * 2, hipMemcpyDeviceToHost));
        printf("| %-44s | %7.3f | | | %s (%zu elements) |\n", "cvt bf16 -> F16 activations [T][K]", cms, memcmp(xcv.data(), xh.data(), n * 2) == 0 ? "matches the host F16 copy" : "DIFFERS from the host F16 copy", n);
        CHECK(hipFree(xc));
    }

    if (f16_only) return 0;
    // ---- double reference: exact Q4_K dequant, double products and sums, double silu ----
    std::vector<int32_t> ids_e(n_rows);
    for (int e = 0; e < E; ++e) for (int p = R.bounds[e]; p < R.bounds[e + 1]; ++p) ids_e[p] = e;
    int32_t * d_e; double * d_ref; CHECK(hipMalloc(&d_e, n_rows * 4)); CHECK(hipMalloc(&d_ref, (size_t) n_rows * M * 8));
    CHECK(hipMemcpy(d_e, ids_e.data(), n_rows * 4, hipMemcpyHostToDevice));
    {
        hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
        CHECK(hipEventRecord(a, st));
        ref_f64_kernel<<<dim3(M / REF_COLS, (n_rows + REF_POS - 1) / REF_POS), REF_COLS * REF_POS, 0, st>>>(D.wg, D.wu, expert_bytes, row_bytes, D.xb, D.ids_src, D.ids_dst, d_e, n_rows, M, K, d_ref);
        CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
        float rms_; CHECK(hipEventElapsedTime(&rms_, a, b));
        printf("| %-44s | %7.1f | | | one pass |\n", "double reference (GPU)", rms_);
    }
    std::vector<double> ref((size_t) n_rows * M); CHECK(hipMemcpy(ref.data(), d_ref, ref.size() * 8, hipMemcpyDeviceToHost));
    double ref_rms = 0, ref_max = 0; for (double v : ref) { ref_rms += v * v; ref_max = std::max(ref_max, fabs(v)); } ref_rms = sqrt(ref_rms / ref.size());
    printf("\nerror vs double reference over %zu outputs (ref rms %.4g, ref maxabs %.4g):\n", ref.size(), ref_rms, ref_max);
    printf("| arm | maxabs | rms | rms / ref rms |\n|---|---:|---:|---:|\n");
    auto errs = [&](const char * name, auto get) {
        double maxabs = 0, se = 0;
        for (size_t i = 0; i < ref.size(); ++i) { const double d = fabs(get(i) - ref[i]); maxabs = std::max(maxabs, d); se += d * d; }
        printf("| %-40s | %.4g | %.4g | %.3e |\n", name, maxabs, sqrt(se / ref.size()), sqrt(se / ref.size()) / ref_rms);
    };
    errs("reference rounded to bf16 (floor for bf16 out)", [&](size_t i) { return (double) bf2f(f2bf((float) ref[i])); });
    errs("reference rounded to F16 (floor for F16 out)", [&](size_t i) { return (double) h2f(f2h((float) ref[i])); });
    errs("legacy / current bf16 kernel, F32 out", [&](size_t i) { return (double) ref_f32[i]; });
    errs("legacy / current bf16 kernel, bf16 out", [&](size_t i) { return (double) bf2f(ref_bf16[i]); });
    errs("gufo RoutedF16GEMMKernel, F16 out", [&](size_t i) { return (double) h2f(gufo_h[i]); });
    errs("new F16 kernel, F32 out", [&](size_t i) { return (double) new_f32[i]; });
    errs("new F16 kernel, bf16 out", [&](size_t i) { return (double) bf2f(new_bf16[i]); });
    CHECK(hipFree(d_ref)); CHECK(hipFree(d_e)); CHECK(hipFree(tiles)); CHECK(hipFree(ctl));
}
