# Generation Benchmarks

These checks are separate from the client harness launchers. They compare Lucebox
generation against a llama.cpp baseline on the same target GGUF, using small
deterministic prompts.

For the paired ragged C1/C4/C8/C16 serving benchmark and the concurrent
HumanEval/GSM8K/Math500/agent suite runner, see
[`concurrency/`](concurrency/README.md).

Use this when you want to know whether a server change affects output quality or
decode speed. Use `harness/clients/` when you want to know whether Codex,
OpenCode, Open WebUI, Pi, and the other clients still work.

## Bench suites (HumanEval, GSM8K, Math500, Agent)

Run standard LLM and agentic benchmarks against a running Lucebox server:

```bash
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080
```

This sends benchmark prompts through the OpenAI-compatible `/v1/chat/completions`
endpoint and reports tok/s, TTFT, and correctness scores.

### Suites

| Suite   | Description                                        | Scoring          |
|---------|----------------------------------------------------|------------------|
| `he`    | HumanEval code-completion prompts (10)             | tok/s only       |
| `gsm`   | GSM8K arithmetic reasoning prompts (10)            | tok/s only       |
| `math`  | Math500 with `\boxed{}` correctness check (10)     | tok/s + accuracy |
| `agent` | Agentic workloads at 2K/8K/24K context (6)         | TTFT + tok/s     |

### Usage

```bash
# All suites (default)
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080

# Only Math500 correctness
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite math

# HumanEval + agent
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --suite he,agent

# Limit to 3 prompts per suite
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --n-sample 3

# Save JSON results
python3 harness/client_test_runner.py bench --url http://127.0.0.1:18080 --json-out /tmp/bench.json
```

### Options

- `--url` (required): Server base URL
- `--suite`: Comma-separated list or `all` (default: `all`)
- `--model`: Model name (default: `luce-dflash`)
- `--n-sample`: Max prompts per suite (default: all in file)
- `--prompts-dir`: Override prompt files directory
- `--json-out`: Write JSON results to this path

### Prompt files

Static JSONL files in `harness/benchmarks/prompts/`:

- `bench_he.jsonl` — HumanEval code-completion
- `bench_gsm.jsonl` — GSM8K arithmetic reasoning
- `bench_math.jsonl` — Math500 with `gold_answer` field
- `bench_agent.jsonl` — Agentic prompts with `bucket` field (2k/8k/24k)

### Correctness

Math500 responses are scored by extracting `\boxed{}` answers and comparing
against gold with normalized math equivalence. Accuracy is reported in the
output but does not gate the exit code.

### Isolated HumanEval scoring from a generation report

For an already-running production server, generate the HumanEval subset with
`generation_benchmark.py run` and score the resulting report separately. The
scorer executes generated code **only** through Bubblewrap: networking is
unshared, host files are not mounted, and the child has CPU, address-space and
process-count limits. A trusted in-sandbox supervisor runs the fixture's gold
test and calls the generated candidate over a narrow JSON RPC; the candidate
has no verdict channel, so its exit status or output cannot mark a case passed.
Each call and shutdown acknowledgement carries a fresh supervisor challenge;
stale, out-of-order, or extra candidate responses fail the score.
Do not use this mode where Bubblewrap's unprivileged user namespace support is
unavailable; it fails closed instead of falling back to host execution.

```bash
python3 harness/benchmarks/generation_benchmark.py run \
  --name k3-he --url http://127.0.0.1:18095/v1 --model kimi-k3 \
  --prompts harness/benchmarks/prompts/bench_he.jsonl \
  --max-tokens 512 --timeout 1500 --json-out /tmp/k3-he-generation.json

python3 harness/client_test_runner.py score-he \
  --generation-report /tmp/k3-he-generation.json \
  --json-out /tmp/k3-he-score.json
```

The static ten-case fixture is a local regression subset, not an official
HumanEval score. A nonzero exit means at least one case failed, the report did
not exactly match the fixture IDs, or the sandbox could not provide a verdict.
The score file is atomically replaced only after every row has a result and
records the hashes of the exact report/fixture bytes it parsed.  Supported
runtime layouts require `/usr/bin/python3`, `/usr`, and `/lib`/`/lib64` as
directories or symlinks into `/usr`; other layouts fail closed.

`client_test_runner.py bench --suite he` is retained unchanged for historical
benchmark parity and directly executes generated code. It is unsafe for
untrusted output and is deprecated for production evaluation; use `score-he`.

---

## Lucebox vs llama.cpp

Run from the repo root on the GPU host:

```bash
harness/benchmarks/run_lucebox_vs_llamacpp.sh
```

The runner starts llama.cpp first, runs the prompt set, stops it, then starts
Lucebox and runs the same prompt set. It is sequential on purpose so a 24 GB
card does not need to hold two copies of the target model.

Common overrides:

```bash
MAX_CTX=65536 MAX_TOKENS=512 harness/benchmarks/run_lucebox_vs_llamacpp.sh
LLAMA_SERVER_BIN=/path/to/llama-server harness/benchmarks/run_lucebox_vs_llamacpp.sh
PROMPTS=/tmp/my_prompts.jsonl harness/benchmarks/run_lucebox_vs_llamacpp.sh
```

Each run writes:

- `llamacpp.json`: raw llama.cpp endpoint results
- `lucebox.json`: raw Lucebox endpoint results
- `compare.json`: machine-readable comparison
- `report.md`: speed and expected-output summary

Prompt files are JSONL. Each line needs `id` and either `prompt` or `messages`.
Optional `expect_contains` and `expect_regex` fields define lightweight accuracy
checks.

---

## DeepSeek 4 exact-context benchmark

`deepseek4/ds4_publication_decode_client.py` runs a deterministic streaming
decode workload and records timing, token counts, and response hashes.
`deepseek4/ds4_context_sweep.py` uses the target model's tokenizer to run the
same workload at exact context lengths.

The AMD q=5 hardware launcher is kept separately under
`harness/qualification/deepseek4/` because it changes GPU performance settings
and assumes a specific two-GPU layout.
