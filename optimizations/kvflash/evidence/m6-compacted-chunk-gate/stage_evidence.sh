#!/bin/bash
W=/home/peppi/Dev/lucebox-builds/wt-m6c
E=$W/optimizations/kvflash/evidence/m6-compacted-chunk-gate
G=/mnt/kimi-k3/kvflash-gate-m6c
B=/mnt/kimi-k3/build
rm -rf $E
mkdir -p $E
cp $G/PROVENANCE.txt $G/cmp.out $E/
cp $B/m6c-ab/{ab_m6.py,kvlib.py,ab_all.sh,ab_cmp.py,ab_run.sh,smoke.sh,stage_evidence.sh} $E/
for arm in main m6c neutral; do
  for i in 1 2 3 4 5; do
    r=${arm}_$i
    mkdir -p $E/$r; cp $G/$r/m6_run.json $G/$r/driver.out $E/$r/
    grep -E 'test compact|kv_bias readback|bias graph check' $G/$r/server.log | sort | uniq -c > $E/$r/bias_lines.txt
  done
done
for r in smoke_neutral2 sanity_beta_neg2; do
  mkdir -p $E/$r; cp $G/$r/m6_run.json $G/$r/driver.out $E/$r/
  grep -E 'test compact|kv_bias readback|bias graph check' $G/$r/server.log | sort | uniq -c > $E/$r/bias_lines.txt
done
python3 -I - > $E/sanity_beta_neg2/oracle.txt <<'EOF'
import json
a = json.load(open("/mnt/kimi-k3/kvflash-gate-m6c/smoke_neutral2/m6_run.json"))
b = json.load(open("/mnt/kimi-k3/kvflash-gate-m6c/sanity_beta_neg2/m6_run.json"))
print("same prompts and flags; arm A hook 20:0 (beta 0), arm B hook 1:-1000 (beta -1000)")
for x, y in zip(a, b):
    n = next((i for i, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q), None)
    print(f"prompt {x['prompt']}: texts identical={x['text'] == y['text']} first differing char={n}")
EOF
cp $B/m6c-fix-red-syntax.log $B/m6c-fix-red-run.keep.log $B/m6c-fix-green-pager.log $B/m6c-fix-build.log $E/ 2>/dev/null
cp $B/m6c-green-pager.log $B/m6c-red-syntax.log $E/
grep -E 'Passed|Failed|Skipped' $B/m6c-fix-unit-full.log > $E/test_server_unit_summary.txt
cat > $E/README.md <<'EOF2'
M6 (Revision 2 item 3) gate after the Codex review fixes, 3090, Qwen3.8-27B-UD-Q4_K_M, --kvflash 8192, --no-kv-cache,
5 x ~18.8K-token prompts, temp 0, 5 interleaved rounds of three arms: main 6f2b1795, m6c (this branch, nothing
compacted), neutral (same binary, LUCE_KVFLASH_TEST_COMPACT=20:0: one chunk marked compacted, K/V untouched,
beta 0). All 5 rounds of raw JSON per arm are included. See PROVENANCE.txt and cmp.out (ab_cmp.py exits non-zero on
any mismatch or missing row). The neutral arm's driver (ab_m6.py) also fails unless every prompt logged a marked chunk,
a readback match and the in-engine graph check (every layer's flash-attention node carries its own kv_bias_<il> when
the bias is on, none when off).
smoke_neutral2 / sanity_beta_neg2: 2 x ~11K-token prompts; the second uses beta -1000. bias_lines.txt names the chunk
that was marked; oracle.txt shows the text changes against beta 0, so the bias reaches attention.
Caches were not dropped (no sudo). The ggml .so md5s of main and m6c differ (separate build trees).
EOF2
