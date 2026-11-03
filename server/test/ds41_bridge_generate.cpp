// DS4.1 next-step agreement with translated memory (preregistration
// ds41-nextstep-0001): teacher-forced metrics plus greedy free-run generation
// and an in-process comparison of the first tool-call block against a full read.
//
//   ds41_bridge_generate MODEL.gguf OUT_PREFIX DIR KEY...
//
// PRIVACY: this binary NEVER writes or prints token ids of session or generated
// content, and never decoded text. Outputs are counts, indices, booleans and
// sha256 hashes of token-id spans. Decoding happens in memory only, to read the
// tool name, and the string is dropped. Session ids stay on the box.
//
// Per session KEY the exporter wrote, under DIR:
//   KEY.ids                       int32 [0, k_start + k_len), BOS at 0
//   KEY.meta.json                 s, k_start, k_len, D_len
//   KEY.{native,trans}.{comp,index}.bin   "DS41ROWS" rows at s (arms Bt / T)
// Prompt = ids[:k_start]; scored tokens = ids[k_start : k_start + k_len].
//
// Arms (run order is always A B Bt T C):
//   A   native full read: prefill ids[:s] -> snapshot -> restore, prefill rest
//       (the probe's chain, so chunk boundaries match a cold prefill)
//   B   native snapshot at s, raw rings zeroed, LUCE_DS41_ENCODER_BOUNDED_REPLAY=1
//   Bt  B with comp/index snapshot tensors overwritten from KEY.native.*.bin
//   T   B with comp/index snapshot tensors overwritten from KEY.trans.*.bin
//   C   fresh BOS + ids[s:k_start], no far memory
// Per arm: teacher-forced NLL/KL/top-1 against A on the scored tokens, free-run
// greedy generation (temperature 0, no oracle, stop at EOS / end of the first
// DSML tool-call block / GEN_MAX), next-step comparison against A's generation,
// and "action_tf": A's generated tokens teacher-forced under this arm, scored
// on A's tool-call block only.
//
// Env: GEN_ARMS (default ABtTBC)  GEN_MAX (3072)  GEN_SKIP_GEN=1 (teacher-forced only)
//      GEN_SHARE_DIR (default <dir of this binary>/share: ds41-lucebox profile data)
// Writes OUT_PREFIX.jsonl (resumable per (key, arm)) and OUT_PREFIX.summary.md.
//
// Model-gated (HIP box, ds41-lucebox profile); not registered in ordinary CI.
#include "ds41_bridge_common.h"
#include "ds41_bridge_generate_util.h"
#include "ds41_bridge_metrics.h"
#include "scoped_env.h"
#include "server/tokenizer.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>

using namespace ds41;
namespace gu = ds41gen;

namespace {

constexpr const char * kFloorEnv = "LUCE_DS41_ENCODER_BOUNDED_REPLAY";
constexpr int kSlot = 0;
const char * const kArmOrder[] = {"A", "B", "Bt", "T", "C"};

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[noreturn]] void die(const std::string & msg, int code = 2) {
    std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
    std::exit(code);
}

std::set<std::string> parse_arms(const char * spec) {
    std::set<std::string> out;
    const std::string s = spec && *spec ? spec : "ABtTBC";
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == 'B' && i + 1 < s.size() && s[i + 1] == 't') { out.insert("Bt"); ++i; }
        else if (std::string("ABTC").find(s[i]) != std::string::npos) out.insert(std::string(1, s[i]));
        else die(std::string("GEN_ARMS: unknown arm character '") + s[i] + "'");
    }
    return out;
}

struct Session {
    std::string key;
    std::vector<int32_t> ids;
    int s = 0, k_start = 0, k_len = 0, d_len = 0;
};

Session load_session(const std::string & dir, const std::string & key) {
    Session se;
    se.key = key;
    const std::string base = dir + "/" + key;
    se.ids = read_ids(base + ".ids");
    std::vector<uint8_t> raw;
    if (!gu::read_file(base + ".meta.json", raw)) die("cannot read " + key + ".meta.json");
    const std::string meta(raw.begin(), raw.end());
    int64_t s = 0, ks = 0, kl = 0, dl = 0;
    if (!gu::json_int(meta, "s", s) || !gu::json_int(meta, "k_start", ks) ||
        !gu::json_int(meta, "k_len", kl) || !gu::json_int(meta, "D_len", dl)) {
        die(key + ": meta.json lacks s / k_start / k_len / D_len");
    }
    se.s = (int) s; se.k_start = (int) ks; se.k_len = (int) kl; se.d_len = (int) dl;
    if (se.ids.empty() || se.ids[0] != 0) die(key + ": ids missing or BOS (0) not at position 0");
    if ((int64_t) se.ids.size() != ks + kl) die(key + ": ids length != k_start + k_len");
    if (se.s <= 0 || (se.s & 1) || se.s > se.k_start) die(key + ": s must be even, > 0 and <= k_start");
    if (se.k_len <= 0) die(key + ": k_len <= 0");
    return se;
}

// Rows for Bt / T: comp and index files of one source, loaded and parsed.
struct RowSet {
    gu::RowsFile comp, index;
};

RowSet load_rows(const std::string & dir, const std::string & key, const char * which) {
    RowSet rs;
    for (int k = 0; k < 2; ++k) {
        const char * part = k == 0 ? "comp" : "index";
        const std::string path = dir + "/" + key + "." + which + "." + part + ".bin";
        std::vector<uint8_t> raw;
        if (!gu::read_file(path, raw)) die("cannot read " + key + "." + which + "." + part + ".bin");
        std::string err;
        if (!gu::parse_rows(raw.data(), raw.size(), k == 0 ? rs.comp : rs.index, err))
            die(key + "." + which + "." + part + ".bin: " + err);
    }
    return rs;
}

struct Overwrite {
    int comp_layers = 0, index_layers = 0;
    int comp_uncovered = 0, index_uncovered = 0;  // snapshot tensors the file did not cover
};

// Overwrites the snapshot's comp / index tensors with `rs`, saving the previous
// bytes of every touched tensor in `backup` (first touch only) so the native
// snapshot can be put back for the next arm. Fails loudly on any shape mismatch.
Overwrite apply_rows(ggml_context * ctx, const RowSet & rs, int s,
                     std::map<ggml_tensor *, std::vector<uint8_t>> & backup) {
    Overwrite ov;
    const auto apply = [&](const gu::RowsFile & f, const char * fmt, const char * prefix, int & n_applied,
                           int & n_uncovered) {
        for (const gu::RowsLayer & l : f.layers) {
            char name[64];
            std::snprintf(name, sizeof name, fmt, (int) l.layer);
            ggml_tensor * t = ggml_get_tensor(ctx, name);
            if (!t) die(std::string(name) + ": snapshot has no such tensor for a layer in the rows file");
            if (t->type != GGML_TYPE_F16 || ggml_n_dims(t) > 2) die(std::string(name) + ": snapshot tensor is not 2D F16");
            std::string err;
            if (!gu::validate_rows_layer(l, s, t->ne[0], t->ne[1], err)) die(std::string(name) + ": " + err);
            if (ggml_nbytes(t) != l.f16.size() * sizeof(uint16_t)) die(std::string(name) + ": byte size mismatch");
            if (!backup.count(t)) backup[t] = tensor_bytes(t);
            ggml_backend_tensor_set(t, l.f16.data(), 0, ggml_nbytes(t));
            ++n_applied;
        }
        for (ggml_tensor * t : snapshot_tensors(ctx, prefix)) {
            int layer = -1;
            std::sscanf(ggml_get_name(t) + std::strlen(prefix), "%d", &layer);
            if (layer < 0 || !f.find((uint32_t) layer)) ++n_uncovered;
        }
    };
    apply(rs.comp, "ds4_snap_comp_kv_%d", "ds4_snap_comp_kv_", ov.comp_layers, ov.comp_uncovered);
    apply(rs.index, "ds4_snap_index_kv_%d", "ds4_snap_index_kv_", ov.index_layers, ov.index_uncovered);
    return ov;
}

void restore_native(std::map<ggml_tensor *, std::vector<uint8_t>> & backup) {
    for (auto & [t, b] : backup) set_tensor_bytes(t, b);
    backup.clear();
}

// One way of running a request: what prompt, how (snapshot restore or fresh),
// and under which floor setting.
struct Arm {
    std::string name;
    std::vector<int32_t> prompt;
    std::vector<int> restore_points;
    bool restore = false;  // restore_and_generate(kSlot) vs generate
    const char * floor = "0";
};

struct ActionScore {
    double nll = NAN, top1 = NAN;
    int n = 0;
};

struct ArmResult {
    ArmScore tf;
    bool gen = false;
    gu::GenInfo info;
    gu::StopKind stop = gu::StopKind::Cap;
    gu::StepCompare cmp;
    ActionScore action;
    double wall_s = 0;
};

// Runs requests for one backend. Greedy: SamplerCfg{} has temp 0 and no
// penalties, so needs_logit_processing() is false and the engine takes the
// argmax (first max wins); no drafter is loaded, so decode is autoregressive.
struct Runner {
    DeepSeek4Backend & be;
    DaemonIO io;
    std::vector<int32_t> open, close, eos;
    int max_gen = 3072;
    const luce::common::Tokenizer & tok;

    GenerateResult run(const Arm & a, GenerateRequest r) {
        r.prompt = a.prompt;
        r.restore_points = a.restore_points;
        r.sampler = SamplerCfg{};
        r.do_sample = false;
        luce_test::ScopedEnvVar floor(kFloorEnv, a.floor);
        return a.restore ? be.restore_and_generate(kSlot, r, io) : be.generate(r, io);
    }

    // Teacher-forced logits over `truth` (full vocab per step).
    bool teacher(const Arm & a, const std::vector<int32_t> & truth, Logits & out) {
        return teacher_force(be, truth, out, [&] {
            GenerateRequest r;
            r.n_gen = (int) truth.size();
            return run(a, r);
        });
    }

    // Free-run greedy generation; the token callback stops it at the end of the
    // first complete tool-call block, the engine at EOS, n_gen at the cap.
    gu::GenInfo generate(const Arm & a) {
        gu::ToolBlockTracker tr(open, close);
        GenerateRequest r;
        r.n_gen = max_gen;
        r.on_token = [&](int32_t t) { return !tr.push(t); };
        const GenerateResult res = run(a, r);
        if (!res.ok() && !tr.span().complete) die("generation failed on arm " + a.name, 1);
        gu::GenInfo g;
        g.tokens = tr.tokens();
        g.span = tr.span();
        if (g.span.complete) {
            // In-memory only: the decoded block is read for the tool name and dropped.
            const std::vector<int32_t> block(g.tokens.begin() + g.span.begin, g.tokens.begin() + g.span.end);
            g.tool_name = gu::extract_tool_name(tok.decode(block));
        }
        return g;
    }

    gu::StopKind stop_of(const gu::GenInfo & g) const {
        const gu::StopKind k = gu::classify_stop(g, eos);
        // The engine stopped before the cap without a block: its own EOS ids won.
        return k == gu::StopKind::Cap && (int) g.tokens.size() < max_gen ? gu::StopKind::Eos : k;
    }

    // A's generated tokens (up to the end of its tool block) forced under arm
    // `a`; NLL / top-1 of the block's tokens only. No full-vocab logits are kept.
    ActionScore action(const Arm & a, const gu::GenInfo & ref) {
        ActionScore sc;
        if (!ref.span.complete) return sc;
        const std::vector<int32_t> forced(ref.tokens.begin(), ref.tokens.begin() + ref.span.end);
        double nll = 0;
        int hits = 0, n = 0;
        be.set_decode_oracle([&](int step, const float * lg, int nv) {
            if (step >= ref.span.begin && step < ref.span.end) {
                double x; bool top1;
                gu::score_step(lg, (size_t) nv, forced[(size_t) step], x, top1);
                nll += x; hits += top1 ? 1 : 0; ++n;
            }
            return forced[(size_t) step];
        });
        GenerateRequest r;
        r.n_gen = (int) forced.size();
        const GenerateResult res = run(a, r);
        be.set_decode_oracle(nullptr);
        if (!res.ok() || n != ref.span.end - ref.span.begin) die("action teacher-forced run failed on arm " + a.name, 1);
        sc.nll = nll / n; sc.top1 = (double) hits / n; sc.n = n;
        return sc;
    }
};

std::string jb(bool v) { return v ? "true" : "false"; }

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUT_PREFIX DIR KEY...\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1], out_prefix = argv[2], dir = argv[3];
    const std::set<std::string> arms = parse_arms(std::getenv("GEN_ARMS"));
    const int max_gen = std::getenv("GEN_MAX") ? std::atoi(std::getenv("GEN_MAX")) : 3072;
    const bool skip_gen = std::getenv("GEN_SKIP_GEN") && std::string(std::getenv("GEN_SKIP_GEN")) == "1";
    if (max_gen <= 0) die("GEN_MAX must be > 0");
    if (std::getenv("LUCE_DS4_DRAFT") && *std::getenv("LUCE_DS4_DRAFT"))
        die("LUCE_DS4_DRAFT is set: DSpark/speculative decode must be off");
    const std::string share = std::getenv("GEN_SHARE_DIR")
        ? std::getenv("GEN_SHARE_DIR")
        : (std::filesystem::absolute(argv[0]).parent_path() / "share").string();

    std::vector<Session> sessions;
    int max_ctx = 0;
    for (int i = 4; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.find_first_of("\",}\\ ") != std::string::npos) die("key has characters unsafe for the jsonl: " + key);
        sessions.push_back(load_session(dir, key));
        const Session & se = sessions.back();
        max_ctx = std::max(max_ctx, se.k_start + std::max(se.k_len, max_gen) + 64);
    }

    // Rows already written (resume): "key|arm".
    std::set<std::string> done;
    {
        std::ifstream in(out_prefix + ".jsonl");
        for (std::string line; std::getline(in, line);)
            done.insert(gu::json_field(line, "key") + "|" + gu::json_field(line, "arm"));
    }
    std::ofstream jsonl(out_prefix + ".jsonl", std::ios::app);

    luce::common::Tokenizer tok;
    if (!tok.load_from_gguf(model.c_str())) die("tokenizer load failed", 1);
    // Marker token ids: "<" / "</" + the DSML token + " calls". The trailing ">" is left
    // out because it can merge with the newline that follows (">\n"). V4.1 spelling,
    // chat_template.cpp append_dsml_tool_calls(v41=true): "<｜DSML｜ calls>" ... "</｜DSML｜ calls>".
    const std::string dsml = "\xEF\xBD\x9C" "DSML" "\xEF\xBD\x9C";
    const std::vector<int32_t> open_m = tok.encode("<" + dsml + " calls");
    const std::vector<int32_t> close_m = tok.encode("</" + dsml + " calls");
    {
        const int32_t dsml_id = tok.token_to_id(dsml);
        const auto has = [&](const std::vector<int32_t> & m) { return std::find(m.begin(), m.end(), dsml_id) != m.end(); };
        if (open_m.size() < 3 || close_m.size() < 3 || open_m == close_m || dsml_id < 0 || !has(open_m) || !has(close_m))
            die("DSML marker tokenization is not what this tool expects (open " + std::to_string(open_m.size()) +
                " tokens, close " + std::to_string(close_m.size()) + ", dsml_id " + std::to_string(dsml_id) + ")", 1);
        std::fprintf(stderr, "[gen] DSML markers: open=%zu tokens, close=%zu tokens\n", open_m.size(), close_m.size());
    }
    std::vector<int32_t> eos;
    if (tok.eos_id() >= 0) eos.push_back(tok.eos_id());
    if (tok.eos_chat_id() >= 0) eos.push_back(tok.eos_chat_id());

    DeepSeek4Backend backend(make_config(model, max_ctx, share));
    if (!backend.init()) die("backend init", 1);
    Runner rn{backend, DaemonIO{}, open_m, close_m, eos, max_gen, tok};

    // Warm the streamed-expert cache (routing near-ties flip with cache state).
    {
        const Session & se = sessions[0];
        GenerateRequest r;
        r.prompt.assign(se.ids.begin(), se.ids.begin() + std::min<size_t>(1024, (size_t) se.k_start));
        r.n_gen = 8;
        for (int i = 0; i < 2; ++i) backend.generate(r, rn.io);
    }

    for (size_t si = 0; si < sessions.size(); ++si) {
        const Session & se = sessions[si];
        const std::vector<int32_t> prompt(se.ids.begin(), se.ids.begin() + se.k_start);
        const std::vector<int32_t> truth(se.ids.begin() + se.k_start, se.ids.end());
        bool pending = false;
        for (const std::string & a : arms) if (!done.count(se.key + "|" + a)) pending = true;
        if (!pending) { std::fprintf(stderr, "[gen] %s: done, skipping\n", se.key.c_str()); continue; }
        std::fprintf(stderr, "[gen] session %zu/%zu s=%d k_start=%d k_len=%d\n", si + 1, sessions.size(), se.s,
                     se.k_start, se.k_len);

        // Diagnostic only: do the marker ids agree with the decoded text on the prompt's
        // earlier tool calls? Counts, never content.
        {
            const std::string text = tok.decode(prompt);
            int text_open = 0;
            for (size_t p = text.find("<" + dsml + " calls"); p != std::string::npos;
                 p = text.find("<" + dsml + " calls", p + 1)) ++text_open;
            std::fprintf(stderr, "[gen] marker check on prompt: open by ids=%d by text=%d\n",
                         gu::count_marker(prompt, open_m), text_open);
        }

        // One native prefill to s, kept as snapshot slot 0 for A's restore and arms B/Bt/T.
        {
            Arm pre{"prefill", std::vector<int32_t>(prompt.begin(), prompt.begin() + se.s), {se.s}, false, "0"};
            GenerateRequest r;
            r.n_gen = 0;
            r.snap_slot = kSlot;
            r.snap_pos = se.s;
            const GenerateResult res = rn.run(pre, r);
            if (!res.ok() || !backend.snapshot_used(kSlot)) die("native prefill to s", 1);
            if (backend.snapshot_ref(kSlot).cur_pos != se.s) die("snapshot cur_pos != s", 1);
        }
        ggml_context * sctx = backend.snapshot_ref(kSlot).ctx;

        const Arm arm_defs[] = {
            {"A",  prompt, {se.s}, true, "0"},
            {"B",  prompt, {se.s}, true, "1"},
            {"Bt", prompt, {se.s}, true, "1"},
            {"T",  prompt, {se.s}, true, "1"},
            {"C",  [&] { std::vector<int32_t> p{prompt[0]}; p.insert(p.end(), prompt.begin() + se.s, prompt.end()); return p; }(),
                   {}, false, "0"},
        };

        Logits logits_a;
        gu::GenInfo gen_a;
        double nll_a = NAN;
        std::map<ggml_tensor *, std::vector<uint8_t>> backup;
        bool rings_zeroed = false;

        for (const Arm & arm : arm_defs) {
            const bool wanted = arms.count(arm.name) && !done.count(se.key + "|" + arm.name);
            if (arm.name != "A" && !wanted) continue;  // A is always computed: it is the reference
            const double t0 = now_s();
            Overwrite ov;

            if (arm.name == "B" || arm.name == "Bt" || arm.name == "T") {
                if (!rings_zeroed) {
                    for (ggml_tensor * t : snapshot_tensors(sctx, "ds4_snap_raw_kv_")) zero_tensor(t);
                    rings_zeroed = true;
                }
                if (arm.name != "B") {
                    const RowSet rs = load_rows(dir, se.key, arm.name == "Bt" ? "native" : "trans");
                    ov = apply_rows(sctx, rs, se.s, backup);
                }
            }

            ArmResult res;
            Logits lg;
            if (!rn.teacher(arm, truth, lg)) die("teacher-forced run on arm " + arm.name, 1);
            if (arm.name == "A") { logits_a = lg; nll_a = score_arm(logits_a, logits_a, truth).nll; }
            res.tf = score_arm(logits_a, lg, truth);

            if (!skip_gen) {
                res.gen = true;
                res.info = arm.name == "A" ? (gen_a = rn.generate(arm)) : rn.generate(arm);
                res.stop = rn.stop_of(res.info);
                res.cmp = gu::compare_next_step(gen_a, res.info);
                res.action = rn.action(arm, gen_a);
            }
            restore_native(backup);
            res.wall_s = now_s() - t0;

            if (!wanted) continue;
            std::string row = "{";
            const auto kv = [&](const char * k, const std::string & v) {
                if (row.size() > 1) row += ",";
                row += "\"" + std::string(k) + "\":" + v;
            };
            const gu::GenInfo & g = res.info;
            kv("key", "\"" + se.key + "\""); kv("arm", "\"" + arm.name + "\"");
            kv("s", std::to_string(se.s)); kv("k_start", std::to_string(se.k_start));
            kv("nll", gu::json_num(res.tf.nll)); kv("kl", gu::json_num(res.tf.kl));
            kv("top1", gu::json_num(res.tf.top1)); kv("nll_a", gu::json_num(nll_a));
            kv("gen", jb(res.gen));
            kv("n_gen", res.gen ? std::to_string(g.tokens.size()) : "null");
            kv("stop", res.gen ? std::string("\"") + gu::stop_name(res.stop) + "\"" : "null");
            kv("has_tool", res.gen ? jb(g.span.complete) : "null");
            kv("a_has_tool", res.gen ? jb(gen_a.span.complete) : "null");
            kv("same_step", res.gen ? jb(res.cmp.same_step) : "null");
            kv("same_tool", res.gen ? jb(res.cmp.same_tool) : "null");
            kv("first_divergence", res.gen ? std::to_string(res.cmp.first_divergence) : "null");
            kv("tool_len", res.gen && g.span.complete ? std::to_string(g.span.end - g.span.begin) : "null");
            kv("sha256_tool_ids", res.gen && g.span.complete
                   ? "\"" + gu::sha256_ids(g.tokens.data() + g.span.begin, (size_t) (g.span.end - g.span.begin)) + "\""
                   : "null");
            kv("sha256_gen_ids", res.gen ? "\"" + gu::sha256_ids(g.tokens.data(), g.tokens.size()) + "\"" : "null");
            kv("action_nll", gu::json_num(res.action.nll)); kv("action_top1", gu::json_num(res.action.top1));
            kv("action_n", std::to_string(res.action.n));
            kv("rows_comp_layers", std::to_string(ov.comp_layers)); kv("rows_index_layers", std::to_string(ov.index_layers));
            kv("snap_comp_uncovered", std::to_string(ov.comp_uncovered));
            kv("snap_index_uncovered", std::to_string(ov.index_uncovered));
            kv("wall_s", gu::json_num(res.wall_s));
            row += "}";
            jsonl << row << "\n";
            jsonl.flush();
            std::fprintf(stderr, "[gen] %s\n", row.c_str());
        }
        backend.snapshot_free(kSlot);
    }

    // Summary over every row in the file (this run and earlier ones). No text.
    {
        std::map<std::string, std::vector<std::map<std::string, std::string>>> by_arm;
        std::ifstream in(out_prefix + ".jsonl");
        static const char * kFields[] = {"nll", "kl", "top1", "nll_a", "gen", "stop", "a_has_tool", "same_step",
                                         "same_tool", "first_divergence", "action_nll", "action_top1"};
        for (std::string line; std::getline(in, line);) {
            std::map<std::string, std::string> f;
            for (const char * k : kFields) f[k] = gu::json_field(line, k);
            by_arm[gu::json_field(line, "arm")].push_back(f);
        }
        const auto num = [](const std::map<std::string, std::string> & f, const char * k, double & v) {
            const std::string & s = f.at(k);
            if (s.empty() || s == "null") return false;
            v = std::atof(s.c_str());
            return true;
        };
        std::ofstream md(out_prefix + ".summary.md");
        const char * hdr =
            "| arm | n | n_gen | A has tool | same_step | same_tool | n identical | first_div med (diverged) | stops tool/eos/cap "
            "| dNLL mean | KL mean | top1 mean | action NLL mean | action top1 mean |\n"
            "|-----|---|-------|------------|-----------|-----------|-------------|--------------------------|--------------------"
            "|-----------|---------|-----------|-----------------|------------------|\n";
        md << hdr;
        std::fputs(hdr, stderr);
        for (const char * name : kArmOrder) {
            const auto it = by_arm.find(name);
            if (it == by_arm.end()) continue;
            std::vector<double> dn, kl, t1, fd, an, at1;
            int n_gen = 0, a_tool = 0, same_step = 0, same_tool = 0, ident = 0, st[3] = {0, 0, 0};
            for (const auto & f : it->second) {
                double v;
                if (num(f, "nll", v)) { double a = 0; dn.push_back(v - (num(f, "nll_a", a) ? a : 0)); }
                if (num(f, "kl", v)) kl.push_back(v);
                if (num(f, "top1", v)) t1.push_back(v);
                if (num(f, "action_nll", v)) an.push_back(v);
                if (num(f, "action_top1", v)) at1.push_back(v);
                if (f.at("gen") != "true") continue;
                ++n_gen;
                a_tool += f.at("a_has_tool") == "true";
                same_step += f.at("same_step") == "true";
                same_tool += f.at("same_tool") == "true";
                if (num(f, "first_divergence", v)) { if (v < 0) ++ident; else fd.push_back(v); }
                st[f.at("stop") == "tool" ? 0 : f.at("stop") == "eos" ? 1 : 2]++;
            }
            char buf[640];
            std::snprintf(buf, sizeof buf,
                          "| %s | %zu | %d | %d | %d/%d | %d/%d | %d | %.0f | %d/%d/%d | %+.4f | %.4f | %.3f | %.4f | %.3f |\n",
                          name, it->second.size(), n_gen, a_tool, same_step, n_gen, same_tool, n_gen, ident, median(fd),
                          st[0], st[1], st[2], mean(dn), mean(kl), mean(t1), mean(an), mean(at1));
            md << buf;
            std::fputs(buf, stderr);
        }
    }
    return 0;
}
