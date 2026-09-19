# qwen4exp PR follow-ups — more tests, less comments (handoff)

State at 2026-09-19 session end. Repo `/home/peppi/Dev/lucebox-qwen4exp`, branch
`feat/qwen4exp-strix-halo` @ `093b80bd` — a **clean 5-commit conventional series**
(squashed from 75) already force-pushed to `fork` (`dusterbloom/lucebox-hub`):

```
093b80bd docs(qwen4exp): backend design and benchmark methodology
8c248f75 fix(harness): score GSM/Math answers correctly and add the planted-recall suite
30895b3e test(qwen4exp): forward smoke and mmb/cuBLAS differentials
d7529cf6 feat(qwen4exp): hand-written Qwen3.8-Flash-Next backend
83ea4b9f feat(ggml-cuda): add Qwen3.8-Flash-Next kernel suite (QSA, MMB bf16 GEMM, HC/GDN/PLE fusion, packed get_rows)
```

Backups of the pre-squash history: tag `backup-qwen4exp-20260919` (`7232988a`,
pushed) and branch `backup-qwen4exp-branch`. Internal handoffs
(`qwen4exp-*.md`) are **untracked / excluded from the PR** by design.

The two gaps vs the upstream bar (`CONTRIBUTING.md`: one concern per PR,
conventional commits, before/after benchmarks, and **tests shipped with the
change**): our branch is ~5% tests vs upstream ~20–100%, and ~12% comment lines.

## Task 1 — add tests

Box: `ssh duster@lucebox4.tail97592a.ts.net`, tree `~/lucebox-qwen4exp`, build
`server/build-hip`. Model: `~/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf`.
Build: `ninja -C server/build-hip <target> -j 24`. Run: `HIP_VISIBLE_DEVICES=1 <bin>`
(or `ctest -R <name>`).

Patterns to mirror:
- `server/test/test_mmb_cublas_diff.cpp` — our GEMM differential (90 lines).
- `server/test/test_ds4_mmid_regression.cpp` — our routed-GEMM regression (168 lines).
- `server/test/test_kvflash*.cpp` — correctness suites with `--niah/--longab`.
- `server/test/bench_paged_attention.cpp` — validates vs a double-precision CPU
  oracle *before* timing.
- CMake wiring: `server/CMakeLists.txt` `add_executable` + `add_test`; the
  `dflash_discover_cppunit_tests(target)` path auto-registers `CppUnitTestFramework`
  cases (see `test_server_unit`).
- Harness (CPU, CI-safe) mirror: `harness/tests/test_ds4_benchmark_tools.py`.

Add, in rough priority:

1. **convert vectorized == scalar (bit-exact).** `convert.cu`:
   `convert_bf16_to_f32_vec` / `convert_f32_to_bf16_vec` vs `convert_unary`.
   Random + tail lengths not divisible by 8; assert bit equality of the uint16/float
   outputs. `server/test/test_convert_vec.cpp`.
2. **`k_get_rows_scalar` == generic `k_get_rows`.** `getrows.cu`, ne00==1 path
   (the QSA top-k sort). Random indices with -1/out-of-range sentinels and sizes
   crossing the 256 boundary. `server/test/test_getrows_scalar.cpp`.
3. **Q5_K bf16 shadow == Q5_K mmb.** Extend `test_mmb_cublas_diff.cpp` with the
   `blk.3.attn_output.weight` shape `[6144,2560]` (Q5_K in the 3-shard model),
   compare cuBLAS-with-shadow vs `mmb_dense<...,45>`.
4. **HC16 fused == unfused.** The `hc_gate_mix` / HC-down `MUL_MAT` reading a
   bf16-marked src1 vs the f32 path (`LLAMA_MMB_HC16=0`). Extend
   `test_ds4_mmid_regression.cpp` style.
5. **Indexer-K cache: chunked-prefill QSA == single-chunk QSA.** The high-value
   one. Same prompt tokenized at chunk 4096/8192/16384 must produce the same
   selected blocks / near-identical logits (`--chunk` env); assert via the FA
   counter (`QWEN4EXP_FA_TELEMETRY=1` → `[fa] graph qsa=12`) and a logits
   distance bound. Can start as a Python/model-level test on the box.
6. **QSA selected == dense on random masks.** Kernel-level: build `q3/K/V/ids`
   with a random complete-block selection and compare `qsa3_attn` vs
   `ggml_flash_attn_ext` over the selected cells.
7. **Mask elision.** Assert that when QSA engages for every full layer the dense
   mask tensor is never read (the FA counter + the `build_full_attn` abort guard).
8. **(done)** `harness/tests/test_math_scoring.py` — 10 case GSM/Math extraction
   units. Extend as needed.

CI: add a job alongside `.github/workflows/speed-profile.yml`; CPU-only scorer
tests can run on hosted runners, GPU tests need a gfx1151 self-hosted runner
(mirror PR 694 "enable model-backed ROCm tests").

## Task 2 — trim comments

Target ≤5% comment lines (currently ~12% of added lines). Keep one-line "why",
delete rationale/archaeology. Move prose into `docs/qwen4exp.md` (already written).

Hot files (added lines):
- `server/src/qwen4exp/qwen4exp_graph.cpp` (1048) — long block comments on QSA,
  HC, mask elision.
- `server/deps/llama.cpp/ggml/src/ggml-cuda/mmb.cu` (1099).
- `server/deps/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` (588) — route/mark comments.
- `server/src/qwen4exp/qwen4exp_loader.cpp` (825).

Delete or shorten: `journey step N` references, the ROCm-10 refutation,
"GLM review" notes, `/tmp` paths, "session", multi-paragraph design rationale,
and any comment that restates the code. Keep only non-obvious invariants
(e.g. "marks are keyed on data pointers; views cannot resolve" → keep one line).

## After changing

1. `git add` the touched files; **amend the matching commit** rather than adding
   new ones: tests → `30895b3e` (test) or `8c248f75` (harness); comments →
   the backend/kernel commits. Easiest: redo the squash:
   ```
   git reset --soft origin/main
   # stage the 5 groups again (paths below) and commit
   ```
   Paths per group: kernels `server/deps/llama.cpp`; backend
   `server/src/qwen4exp server/src/common/{backend_factory.cpp,model_capabilities.h}
   server/src/server/{chat_template.cpp,disk_prefix_cache.h,model_card.cpp}
   server/docs/ENVIRONMENT.md server/CMakeLists.txt`; tests `server/test`; harness
   `harness`; docs `docs/qwen4exp.md`.
2. Force-push with lease: `git push --force-with-lease fork
   feat/qwen4exp-strix-halo:feat/qwen4exp-strix-halo`.
3. Re-verify: `ruff check harness server/scripts` (our files clean),
   `ninja -C server/build-hip test_server_unit <new tests>`,
   `HIP_VISIBLE_DEVICES=1 ctest -R "kvflash|PagedKv|AdaptiveSpec|SpecAccept"`,
   the planted gate and harness suites (HE/GSM/Math/recall), and min-of-8
   prefill (`bash /tmp/bench_model.sh <model>` style).
4. Open PR(s) with the before/after table: min-of-8 prefill 16,366 924→988 t/s,
   64K 912→926, 13.6k ~960; HE 10/10, GSM 10/10, Math 9/10, recall 2/2; plus the
   methodology (min-of-N prefill-only; planted gate; harness suites).

## Reference numbers / commands (this box)

- Best env: `QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5 DFLASH_MMB_SHADOW=1 LLAMA_MMB_HC16=2`.
- Server: `server/build-hip/dflash_server <model> --host 127.0.0.1 --port 8700
  --target-device hip:0 --max-ctx 40000 --chunk 16384`.
- Harness: `python3 harness/client_test_runner.py bench --url http://127.0.0.1:8700
  --suite he,gsm,math,recall --model dflash`.
- Tier-2 gates already pass: `test_server_unit` kvflash/spec/paged subsets,
  `bench_paged_attention` (oracle max-abs-error ~5e-6–1e-5).
