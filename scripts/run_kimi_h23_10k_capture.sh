#!/usr/bin/env bash
# Crash-bounded capture stage for the earned H23 10K all-layer calibration.
# Run the long mode only through scripts/gpu_lease.sh after reviewing preflight.
set -euo pipefail

mode="${1:-preflight}"
[[ "$mode" == preflight || "$mode" == status || "$mode" == benchmark || "$mode" == capture || "$mode" == merge ]] || {
    echo "usage: $0 preflight|status|benchmark|capture|merge" >&2
    exit 2
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KIMI_H23_10K_BUILD_DIR:-$repo_dir/server/build-k3-p20-cuda126b}"
model="${KIMI_H23_10K_MODEL:-/mnt/kimi-k3/models/unsloth-Kimi-K3-GGUF/UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00014.gguf}"
corpus="${KIMI_H23_10K_CORPUS:-/mnt/kimi-k3/captures/kimi_panel_smoke.jsonl}"
root="${KIMI_H23_10K_ROOT:-/mnt/kimi-k3/captures/kimi-h23-all-layer-10000-chunked-v1}"
merged="${KIMI_H23_10K_MERGED_ROOT:-$root/merged}"
total_tokens="${KIMI_H23_10K_TOKENS:-10000}"
rows_per_chunk="${KIMI_H23_10K_ROWS_PER_CHUNK:-8}"
max_context="${KIMI_H23_10K_MAX_CONTEXT:-512}"
chunk_tokens="${KIMI_H23_10K_FORWARD_CHUNK:-128}"
gpu="${KIMI_H23_10K_GPU:-0}"
tool="$repo_dir/scripts/kimi_h23_capture_chunks.py"

for required in "$model" "$corpus" "$tool"; do
    [[ -f "$required" ]] || { echo "missing H23 10K input: $required" >&2; exit 1; }
done

mkdir -p "$root"
python3 "$tool" prepare --corpus "$corpus" --root "$root" \
    --total-tokens "$total_tokens" --rows-per-chunk "$rows_per_chunk" \
    >"$root/preflight-plan.json"
python3 "$tool" status --root "$root" >"$root/status.json"

if [[ "$mode" == status ]]; then
    cat "$root/status.json"
    exit 0
fi

free_kib="$(df --output=avail /mnt/kimi-k3 | tail -n 1 | tr -d ' ')"
minimum_free_kib=$((350 * 1024 * 1024))
cat <<EOF
H23 10K capture preflight
  root:                    $root
  model:                   $model
  corpus:                  $corpus
  target tokens:           $total_tokens
  rows per crash chunk:    $rows_per_chunk
  forward chunk:           $chunk_tokens
  expected capture bytes:  6.25 GiB (12.50 GiB while chunks + merge coexist)
  full pipeline estimate:  315 GiB; enforced free-space reserve: 350 GiB
  free space now:          $((free_kib / 1024 / 1024)) GiB
  device cache / threads:  ${KIMI_H23_DEVICE_CACHE_MB:-2048} MiB / ${KIMI_H23_CPU_THREADS:-12}
EOF
if (( free_kib < minimum_free_kib )); then
    echo "insufficient space for the complete H23 10K pipeline reserve" >&2
    exit 1
fi
if [[ "$mode" == preflight ]]; then
    cat "$root/status.json"
    exit 0
fi

if [[ "$mode" == merge ]]; then
    python3 "$tool" merge --root "$root" --output "$merged" \
        >"$root/merge-manifest.stdout.json"
    echo "H23 10K merged capture complete: $merged"
    exit 0
fi

if [[ "$mode" == benchmark ]]; then
    [[ "${KIMI_H23_ALLOW_BENCHMARK:-0}" == 1 ]] || {
        echo "refusing one-chunk benchmark without KIMI_H23_ALLOW_BENCHMARK=1" >&2
        exit 1
    }
else
    [[ "${KIMI_H23_ALLOW_LONG_RUN:-0}" == 1 ]] || {
        echo "refusing long capture without KIMI_H23_ALLOW_LONG_RUN=1" >&2
        echo "run only after approval: scripts/gpu_lease.sh run H23-10K -- $0 capture" >&2
        exit 1
    }
fi

CCACHE_DIR=/tmp/kimi-p20-ccache cmake --build "$build_dir" -j4 \
    --target capture_kimi_k3_panel
mkdir -p "$root/chunks" "$root/telemetry" "$root/rejected"

while :; do
    python3 "$tool" status --root "$root" >"$root/status.json"
    if [[ "$(jq -r .complete "$root/status.json")" == true ]]; then
        break
    fi
    invalid="$(jq -r '.invalid_next.directory // empty' "$root/status.json")"
    if [[ -n "$invalid" ]]; then
        stamp="$(date -u +%Y%m%dT%H%M%SZ)"
        destination="$root/rejected/$(basename "$invalid").$stamp"
        echo "quarantining incomplete chunk: $invalid -> $destination" >&2
        mv "$invalid" "$destination"
        python3 "$tool" status --root "$root" >"$root/status.json"
    fi
    index="$(jq -r '.next_chunk.index' "$root/status.json")"
    remaining="$(jq -r '.remaining_tokens' "$root/status.json")"
    chunk_corpus_rel="$(jq -r '.next_chunk.corpus_relative_path' "$root/status.json")"
    chunk_corpus="$root/$chunk_corpus_rel"
    output="$root/chunks/chunk-$(printf '%04d' "$index")"
    prefix="$root/telemetry/chunk-$(printf '%04d' "$index")"
    [[ ! -e "$output" ]] || { echo "unexpected chunk output exists: $output" >&2; exit 1; }
    echo "[h23-10k] chunk=$index remaining=$remaining corpus=$chunk_corpus"
    python3 "$repo_dir/scripts/run_with_telemetry.py" \
        --output-json "$prefix.telemetry.json" \
        --samples-csv "$prefix.telemetry.csv" \
        --stdout "$prefix.stdout.log" --stderr "$prefix.stderr.log" \
        --mount-path /mnt/kimi-k3 --gpu "$gpu" --interval 2 -- \
        env \
            DFLASH_KIMI_CPU_THREADS="${KIMI_H23_CPU_THREADS:-12}" \
            DFLASH_MOE_NVME_DIRECT=on \
            DFLASH_MOE_NVME_DEVICE_CACHE_MB="${KIMI_H23_DEVICE_CACHE_MB:-2048}" \
            DFLASH_KIMI_MMAP_DROP_PAGES=1 \
            DFLASH_KIMI_MOE_CORE_OFFLOAD=0 \
            "$build_dir/capture_kimi_k3_panel" \
                "$model" "$chunk_corpus" "$output" \
                "$gpu" all "$remaining" "$max_context" "$chunk_tokens" cpu
    python3 "$tool" validate-chunk --root "$root" --index "$index" \
        >"$prefix.validation.json"
    if [[ "$mode" == benchmark ]]; then
        python3 "$tool" status --root "$root" >"$root/status.json"
        echo "H23 10K one-chunk benchmark complete; long capture remains stopped"
        cat "$root/status.json"
        exit 0
    fi
done

python3 "$tool" merge --root "$root" --output "$merged" \
    >"$root/merge-manifest.stdout.json"
echo "H23 10K chunked capture and merge complete: $merged"
