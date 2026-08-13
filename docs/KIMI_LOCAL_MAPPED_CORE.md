# Local Kimi K3 mapped-core execution

The complete fourteen-shard Unsloth Kimi K3 checkpoint now executes on the
local 64 GB Windows / 24 GB RTX 3090 machine without moving the routed expert
bank or changing its bytes.

The opt-in placement is:

```text
read-only GGUF mappings -> CPU KDA, MLA, shared experts, joins, output head
local SN850X            -> exact IQ1_S routed-expert reads
RTX 3090                -> exact routed-expert evaluation and bounded cache
```

This is not the ordinary Luce Spark implementation. Spark's calibrated hot/cold
cache targets Laguna and Qwen. Kimi reuses the lower-level model-neutral NVMe
streamer already qualified for its 896 experts, 16 routes, and latent SiTU
geometry.

## Reproduction

```bash
KIMI_EXACT_CORE=cpu \
KIMI_EXACT_OUTPUT_DIR=/mnt/kimi-k3/results/kimi-exact-baseline-local-cpu \
DFLASH_MOE_NVME_DEVICE_CACHE_MB=16384 \
DFLASH_KIMI_CPU_THREADS=18 \
scripts/run_kimi_exact_baseline.sh
```

The command verifies all model-shard checksums, uses a four-job build cap,
runs the frozen prompt twice under telemetry, and requires byte-identical
behavior and full-logit traces.

## Measured result

Both complete runs passed. They produced prompt IDs
`18805 308 799 5624 12524`, output IDs `198 1587 57195 422`, and the text
prefix `According to all known laws\nof aviation,`. All eight scored
full-vocabulary logit rows are byte-identical across runs. Maximum absolute
logit difference and teacher-to-candidate probability divergence are zero;
top-choice agreement is 8/8.

The first run measured 107.905 seconds for five-token prefill and 66.605
seconds for four-token decode. It moved 55.978 GiB of expert payload at 4.404
GiB/s active I/O with zero errors and timeouts. Peak graphics memory was
17,358 MiB and process high-water resident memory was 23,416,632 KiB. This is
a correctness substrate, not a fast deployment: the CPU core makes decode
about 0.06 token/s.

## Numerical boundary

The unchanged CUDA prefix was re-run at the registered 34-token batch shape
and matched the existing capture byte-for-byte. The mapped CPU prefix also
repeats byte-for-byte, and mapping is byte-identical to the copied CPU weight
path. CPU and CUDA arithmetic are not bit-identical to each other: their
layer-one latent mean cosine is 0.996000 and routed expert-set agreement is
0.990809 on the control sequence.

Therefore H16 must use the new repeated CPU-core result as its local exact
teacher and compare every intervention on that same backend. It must not mix
terminal logits from the CPU core with the earlier CUDA layer-boundary teacher.
