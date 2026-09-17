// E0 parity harness for the Qwen->DS4 cross-family context bridge spike
// (luce_box lucebox_rnd/KV_translator_qwen_ds4, ADR-0001).
//
// Validates the repair protocol itself before any translator exists: a
// native DS4 checkpoint saved at a cut, restored, and suffix-repaired must
// match uninterrupted native prefill.
//
// Model-gated, not registered in ordinary CI. Self-skips without a model.
//
//   DS4_TEST_MODEL=/opt/models/deepseek-v4-flash.gguf \
//   DS4_PARITY_LENGTHS=2048,8192[,32768,131072] \
//   DS4_PARITY_CLASSES=chat,code,tools,longdoc,needle,ident \
//   DS4_PARITY_MODES=aligned,seal,zero \
//   test_ds4_snapshot_parity
//
// Modes mirror the bridge checkpoint modes:
//   aligned  cut = largest 128-token boundary <= L-128  (PREFIX_REPAIR_R)
//   seal     cut = L-1                                  (DIRECT_SEAL)
//   zero     cut = L                                    (DIRECT_ZERO)
//
// Per case:
//   A: generate(n_gen=0, snap(0, cut)); snapshot_save(1)      -> reference @L
//      generate(n_gen=N, greedy)                               -> continuation A
//   B: restore_and_generate(0, n_gen=0); snapshot_save(2)      -> repaired @L
//      compare snapshots (1) vs (2) tensor-by-tensor
//      restore_and_generate(0, n_gen=N, greedy)                -> continuation B
//      compare continuations A vs B
//
// Gates: snapshot positions, greedy continuation equality, no non-finite
// values, exact I32 meta equality, per-tensor float diffs within tolerance
// (chunk-boundary F16 accumulation dominates residuals; defaults are loose,
// tighten with DS4_PARITY_ABS_TOL / DS4_PARITY_REL_TOL).

#include "deepseek4/deepseek4_backend.h"
#include "server/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace dflash::common;

namespace {

std::string env_str(const char * name, const std::string & fallback) {
    const char * v = std::getenv(name);
    return v && v[0] ? std::string(v) : fallback;
}

int env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    return v && v[0] ? std::atoi(v) : fallback;
}

float env_float(const char * name, float fallback) {
    const char * v = std::getenv(name);
    return v && v[0] ? (float) std::atof(v) : fallback;
}

std::vector<std::string> env_list(const char * name,
                                  const std::string & fallback) {
    std::string raw = env_str(name, fallback);
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= raw.size()) {
        const size_t comma = raw.find(',', begin);
        const std::string item = raw.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        if (!item.empty()) out.push_back(item);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return out;
}

// ─── Deterministic prompt-class text ────────────────────────────────────

const char * kChat[] = {"summarize", "the", "quarterly", "report", "and",
    "highlight", "risks", "in", "the", "supply", "chain", "section",
    "please", "answer", "concisely", "with", "bullet", "points", "first",
    "consider", "the", "constraints", "before", "proposing", "anything"};
const char * kCode[] = {"def", "process_items(items,", "batch_size=64):",
    "results", "=", "[]", "for", "i", "in", "range(0,", "len(items),",
    "batch_size):", "chunk", "=", "items[i:i+batch_size]", "results.extend(",
    "transform(chunk))", "return", "results", "#", "note:", "thread-safe",
    "via", "lock", "around", "shared", "state"};
const char * kTools[] = {"<tool_call>", "name:", "search_flights", "arguments:",
    "{\"origin\":", "\"JFK\",", "\"destination\":", "\"NRT\",", "\"date\":",
    "\"2026-10-14\"}", "</tool_call>", "the", "assistant", "should", "call",
    "the", "weather", "tool", "next", "with", "city", "tokyo", "and", "units",
    "metric", "then", "summarize", "both", "results"};
const char * kLongDoc[] = {"the", "committee", "reviewed", "the", "proposal",
    "in", "detail", "and", "asked", "for", "revisions", "to", "section",
    "four", "covering", "budget", "assumptions", "meanwhile", "the",
    "engineering", "team", "continued", "integration", "work", "on", "the",
    "planned", "release", "train", "without", "interruption"};
const char * kIdent[] = {"commit", "9f3a1c7e2b8d4061a", "refactors", "the",
    "loader", "ticket", "LUCE-4471", "tracks", "the", "regression", "sha256",
    "e2b9...", "key", "0x7f31a4", "build", "#88213", "passed", "endpoint",
    "/v1/bridge/adopt", "model", "ds4-flash-roc-mfpx", "host", "lucebox4"};

std::string build_text(const std::string & cls, std::mt19937_64 & rng,
                       size_t target_chars) {
    static const char * needle = " The maintenance access code for bay 4 is "
                                 "BRIDGE-7731-VERA. Remember it for later. ";
    const char ** bank = kLongDoc;
    size_t bank_n = 0;
    if (cls == "chat")         { bank = kChat;    bank_n = std::size(kChat); }
    else if (cls == "code")    { bank = kCode;    bank_n = std::size(kCode); }
    else if (cls == "tools")   { bank = kTools;   bank_n = std::size(kTools); }
    else if (cls == "needle")  { bank = kLongDoc; bank_n = std::size(kLongDoc); }
    else if (cls == "ident")   { bank = kIdent;   bank_n = std::size(kIdent); }
    else if (cls == "longdoc") { bank = kLongDoc; bank_n = std::size(kLongDoc); }

    std::string text;
    text.reserve(target_chars + 256);
    size_t since_needle = 0;
    while (text.size() < target_chars) {
        if (cls == "needle" && since_needle > target_chars / 3) {
            text += needle;
            since_needle = 0;
            continue;
        }
        text += bank[rng() % bank_n];
        text += ((rng() % 8 == 0) ? '\n' : ' ');
        since_needle += 8;
    }
    if (cls == "needle") text += needle;
    if (cls != "code") text = "User request follows. " + text;
    return text;
}

std::vector<int32_t> build_prompt(const Tokenizer & tok,
                                  const std::string & cls, int length,
                                  uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int attempt = 0; attempt < 4; ++attempt) {
        const std::string text = build_text(cls, rng,
                                            (size_t) length * (6 + 2 * attempt));
        std::vector<int32_t> tokens = tok.encode(text);
        if ((int) tokens.size() >= length) {
            tokens.resize(length);
            return tokens;
        }
    }
    return {};
}

// ─── Snapshot comparison ────────────────────────────────────────────────

struct TensorDiff {
    std::string name;
    double max_abs = 0.0;
    double ref_max = 0.0;   // largest |reference| value
    long nonfinite = 0;
};

struct SnapshotDiff {
    bool comparable = false;
    std::string error;
    std::vector<TensorDiff> tensors;
    long i32_mismatches = 0;
    std::string worst_name;
    double worst_abs = 0.0;
    double worst_rel = 0.0;
};

bool tensor_is_float(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 ||
           type == GGML_TYPE_BF16;
}

double read_element_as_double(const uint8_t * data, enum ggml_type type,
                              size_t element_index, bool & nonfinite) {
    if (type == GGML_TYPE_F32) {
        const float v = ((const float *) data)[element_index];
        nonfinite |= !std::isfinite(v);
        return v;
    }
    if (type == GGML_TYPE_F16) {
        const float v = ggml_fp16_to_fp32(((const ggml_fp16_t *) data)[element_index]);
        nonfinite |= !std::isfinite(v);
        return v;
    }
    if (type == GGML_TYPE_BF16) {
        const uint16_t raw = ((const uint16_t *) data)[element_index];
        const uint32_t bits = ((uint32_t) raw) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        nonfinite |= !std::isfinite(f);
        return f;
    }
    return (double) ((const int32_t *) data)[element_index];
}

SnapshotDiff compare_snapshots(const ModelBackend::SnapshotRef & ref,
                               const ModelBackend::SnapshotRef & rep) {
    SnapshotDiff out;
    if (!ref.ctx || !rep.ctx) {
        out.error = "snapshot context missing";
        return out;
    }
    std::map<std::string, ggml_tensor *> a, b;
    for (ggml_tensor * t = ggml_get_first_tensor(ref.ctx); t;
         t = ggml_get_next_tensor(ref.ctx, t))
        a[ggml_get_name(t) ? ggml_get_name(t) : ""] = t;
    for (ggml_tensor * t = ggml_get_first_tensor(rep.ctx); t;
         t = ggml_get_next_tensor(rep.ctx, t))
        b[ggml_get_name(t) ? ggml_get_name(t) : ""] = t;
    if (a.size() != b.size()) {
        out.error = "tensor count mismatch: " + std::to_string(a.size()) +
                    " vs " + std::to_string(b.size());
        return out;
    }
    out.comparable = true;

    std::vector<uint8_t> ra, rb;
    for (const auto & [name, ta] : a) {
        auto it = b.find(name);
        if (it == b.end() || !it->second) {
            out.error = "tensor missing in repaired snapshot: " + name;
            out.comparable = false;
            return out;
        }
        ggml_tensor * tb = it->second;
        if (ta->type != tb->type) {
            out.error = "type mismatch for " + name;
            out.comparable = false;
            return out;
        }
        size_t na = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) na *= (size_t) ta->ne[d];
        size_t nb = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) nb *= (size_t) tb->ne[d];
        if (na != nb) {
            out.error = "shape mismatch for " + name;
            out.comparable = false;
            return out;
        }
        const size_t bytes = ggml_nbytes(ta);
        ra.resize(bytes);
        rb.resize(bytes);
        ggml_backend_tensor_get(ta, ra.data(), 0, bytes);
        ggml_backend_tensor_get(tb, rb.data(), 0, bytes);

        TensorDiff diff;
        diff.name = name;
        if (tensor_is_float(ta->type)) {
            bool nonfinite = false;
            for (size_t i = 0; i < na; ++i) {
                const double va = read_element_as_double(
                    ra.data(), ta->type, i, nonfinite);
                const double vb = read_element_as_double(
                    rb.data(), tb->type, i, nonfinite);
                const double d = std::fabs(va - vb);
                if (d > diff.max_abs) diff.max_abs = d;
                const double m = std::fabs(va);
                if (m > diff.ref_max) diff.ref_max = m;
            }
            diff.nonfinite = nonfinite ? 1 : 0;
        } else if (ta->type == GGML_TYPE_I32) {
            const int32_t * ia = (const int32_t *) ra.data();
            const int32_t * ib = (const int32_t *) rb.data();
            for (size_t i = 0; i < na; ++i)
                if (ia[i] != ib[i]) ++diff.nonfinite;  // reused as mismatch count
            out.i32_mismatches += diff.nonfinite;
        } else if (std::memcmp(ra.data(), rb.data(), bytes) != 0) {
            ++diff.nonfinite;
        }
        out.tensors.push_back(diff);
    }

    for (const auto & d : out.tensors) {
        const double scale = std::max<double>(d.ref_max, 1.0);
        const double rel = d.max_abs / scale;
        if (d.max_abs > out.worst_abs) {
            out.worst_abs = d.max_abs;
            out.worst_name = d.name;
        }
        if (rel > out.worst_rel) out.worst_rel = rel;
    }
    return out;
}

}  // namespace

int main() {
    const char * model_env = std::getenv("DS4_TEST_MODEL");
    if (!model_env || !model_env[0]) {
        std::fprintf(stderr, "SKIP: set DS4_TEST_MODEL to a DS4 *.gguf to run "
                             "this test\n");
        return 0;
    }

    const std::vector<int> lengths = [] {
        std::vector<int> out;
        for (const auto & s : env_list("DS4_PARITY_LENGTHS", "2048,8192"))
            out.push_back(std::atoi(s.c_str()));
        return out;
    }();
    const auto classes = env_list(
        "DS4_PARITY_CLASSES", "chat,code,tools,longdoc,needle,ident");
    const auto modes = env_list("DS4_PARITY_MODES", "aligned,seal,zero");
    const int n_gen = env_int("DS4_PARITY_NGEN", 64);
    const uint64_t seed = (uint64_t) env_int("DS4_PARITY_SEED", 7);
    const float abs_tol = env_float("DS4_PARITY_ABS_TOL", 0.10f);
    const float rel_tol = env_float("DS4_PARITY_REL_TOL", 1e-2f);

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model_env)) {
        std::fprintf(stderr, "SKIP: tokenizer failed to load from %s\n",
                     model_env);
        return 0;
    }

    const int max_length = *std::max_element(lengths.begin(), lengths.end());
    DeepSeek4BackendConfig config;
    config.model_path = model_env;
#if defined(DFLASH27B_BACKEND_HIP)
    config.device.backend = PlacementBackend::Hip;
#else
    config.device.backend = PlacementBackend::Cuda;
#endif
    config.device.gpu = env_int("DS4_PARITY_GPU", 0);
    config.paged_attention = false;
    config.max_concurrency = 1;
    config.prefill_mode = PrefillAttentionMode::Exact;
    config.chunk = env_int("DS4_PARITY_CHUNK", 512);
    config.max_ctx = max_length + n_gen + 256;

    DeepSeek4Backend backend(config);
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init failed\n");
        return 1;
    }

    DaemonIO io;
    int failures = 0;
    int cases = 0;

    for (int length : lengths) {
        if (length < 256) {
            std::fprintf(stderr, "[E0] skip L=%d (<256)\n", length);
            continue;
        }
        for (const auto & cls : classes) {
            const std::vector<int32_t> prompt =
                build_prompt(tokenizer, cls, length, seed + (uint64_t) length);
            if ((int) prompt.size() != length) {
                std::fprintf(stderr,
                             "FAIL: class=%s L=%d produced %zu tokens\n",
                             cls.c_str(), length, prompt.size());
                ++failures;
                continue;
            }
            for (const auto & mode : modes) {
                int cut;
                if (mode == "aligned") {
                    cut = (length - 128) & ~127;
                } else if (mode == "seal") {
                    cut = length - 1;
                } else if (mode == "zero") {
                    cut = length;
                } else {
                    continue;
                }
                ++cases;
                std::fprintf(stderr,
                             "[E0] L=%d class=%s mode=%s cut=%d R=%d\n",
                             length, cls.c_str(), mode.c_str(), cut,
                             length - cut);

                bool ok = true;
                const char * why = "";
                const auto case_t0 = std::chrono::steady_clock::now();

                // A: reference run — checkpoint at cut, state at L, continuation.
                GenerateRequest req_a;
                req_a.prompt = prompt;
                req_a.n_gen = 0;
                req_a.snap_slot = 0;
                req_a.snap_pos = cut;
                if (!backend.generate(req_a, io).ok() ||
                    !backend.snapshot_used(0) ||
                    backend.snapshot_cur_pos(0) != cut) {
                    ok = false;
                    why = "reference checkpoint at cut failed";
                }
                if (ok && !backend.snapshot_save(1)) {
                    ok = false;
                    why = "reference snapshot_save(1) at L failed";
                }
                GenerateRequest req_gen;
                req_gen.prompt = prompt;
                req_gen.n_gen = n_gen;
                std::vector<int32_t> cont_a;
                if (ok) {
                    // Decode straight from the reference state@L snapshot: an
                    // exact full-prompt hit, so no prompt pass is repeated.
                    // Continuations A and B then differ only in how state@L was
                    // produced (whole-prompt prefill vs restore+suffix repair).
                    GenerateResult res_a = backend.restore_and_generate(1, req_gen, io);
                    cont_a = res_a.tokens;
                    if (!res_a.ok() || (int) cont_a.size() != n_gen) {
                        ok = false;
                        why = "reference continuation failed";
                    }
                }

                // B: restore + suffix repair — state at L, then continuation.
                GenerateResult cont_b;
                std::vector<int32_t> cont_b_tokens;
                if (ok) {
                    GenerateRequest req_b0;
                    req_b0.prompt = prompt;
                    req_b0.n_gen = 0;
                    if (!backend.restore_and_generate(0, req_b0, io).ok() ||
                        !backend.snapshot_save(2)) {
                        ok = false;
                        why = "restore + suffix repair failed";
                    }
                }
                SnapshotDiff diff;
                if (ok) {
                    diff = compare_snapshots(backend.snapshot_ref(1),
                                             backend.snapshot_ref(2));
                    if (!diff.comparable) {
                        ok = false;
                        why = diff.error.c_str();
                    } else if (diff.i32_mismatches > 0) {
                        ok = false;
                        why = "integer meta/row-count tensors diverged";
                    } else {
                        long nonfinite = 0;
                        for (const auto & d : diff.tensors)
                            nonfinite += d.nonfinite;
                        if (nonfinite > 0) {
                            ok = false;
                            why = "non-finite values in repaired state";
                        }
                    }
                }
                if (ok) {
                    GenerateRequest req_b;
                    req_b.prompt = prompt;
                    req_b.n_gen = n_gen;
                    cont_b = backend.restore_and_generate(0, req_b, io);
                    cont_b_tokens = cont_b.tokens;
                    if (!cont_b.ok() || (int) cont_b_tokens.size() != n_gen) {
                        ok = false;
                        why = "repaired continuation failed";
                    } else if (cont_b_tokens != cont_a) {
                        ok = false;
                        why = "continuation divergence";
                        for (size_t i = 0; i < cont_a.size() && i < cont_b_tokens.size(); ++i) {
                            if (cont_a[i] != cont_b_tokens[i]) {
                                std::fprintf(stderr,
                                    "     first divergence at +%zu: %d vs %d\n",
                                    i, cont_a[i], cont_b_tokens[i]);
                                break;
                            }
                        }
                    }
                }
                if (ok && diff.comparable) {
                    double worst_abs = 0.0, worst_rel = 0.0;
                    std::string worst_name;
                    for (const auto & d : diff.tensors) {
                        const double scale = std::max<double>(d.ref_max, 1.0);
                        const double rel = d.max_abs / scale;
                        if (rel > worst_rel) {
                            worst_rel = rel;
                            worst_abs = d.max_abs;
                            worst_name = d.name;
                        }
                    }
                    std::fprintf(stderr,
                                 "     state: %zu tensors, worst=%s "
                                 "abs=%.3g rel=%.3g (tol %.3g/%.3g)\n",
                                 diff.tensors.size(), worst_name.c_str(),
                                 worst_abs, worst_rel, abs_tol, rel_tol);
                    if (worst_abs > abs_tol && worst_rel > rel_tol) {
                        ok = false;
                        why = "state diff exceeds tolerance";
                    }
                }

                const double case_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - case_t0).count();
                if (ok) {
                    std::fprintf(stderr, "     PASS (%.1fs)\n", case_s);
                } else {
                    ++failures;
                    std::fprintf(stderr, "     FAIL: %s (%.1fs)\n", why, case_s);
                }
                backend.snapshot_free(0);
                backend.snapshot_free(1);
                backend.snapshot_free(2);
            }
        }
    }

    std::fprintf(stderr, "[E0] %d cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
