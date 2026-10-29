// DS4.1 bridge probe (phase P0): the upper bound for any bridge that
// translates only DS4.1's global KV (compressed + index rows).
//
//   ds41_bridge_probe MODEL.gguf SHARE_DIR OUT_PREFIX DOC.ids [DOC.ids ...]
//
// Each DOC.ids is int32 [BOS, D (L tokens), C (K tokens)]. For every doc and
// replay length R (env PROBE_R, default "128,256"), with s = L - R made even:
//   A native      cold prefill of BOS+D (chunks start at every s, like a prefix
//                 cache with checkpoints), then teacher-forced decode over C.
//   B oracle      native snapshot at s, encoder bounded replay (all-layer raw
//                 floor at s, snapshot rings zeroed), prefill D[s:], C.
//   C truncated   fresh sequence of BOS + D[s:], C: no far context. The lower
//                 bound a bridge must beat.
//   D zero-global B with comp_kv and index_kv zeroed: should be ~C or worse.
// Spec/DSpark decode is off (no drafter), so logits are deterministic.
// Writes OUT_PREFIX.jsonl (one row per doc, R, arm) and OUT_PREFIX.summary.md.
//
// Env: PROBE_R=128,256  PROBE_ARMS=ABCD  PROBE_K=128
//
// Model-gated (lucebox6, ds41-lucebox profile); not registered in ordinary CI.
#include "ds41_bridge_common.h"
#include "ds41_bridge_metrics.h"
#include "scoped_env.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <tuple>

using namespace ds41;

namespace {

constexpr const char * kFloorEnv = "LUCE_DS41_ENCODER_BOUNDED_REPLAY";

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<int> parse_ints(const char * env, const char * dflt) {
    const char * v = std::getenv(env);
    std::stringstream ss(v && *v ? v : dflt);
    std::vector<int> out;
    for (std::string item; std::getline(ss, item, ',');) out.push_back(std::atoi(item.c_str()));
    return out;
}

struct Row {
    std::string doc;
    int L = 0, R = 0;
    char arm = 'A';
    double nll = 0, kl = 0, top1 = 0, nll_a = 0, wall_s = 0;
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s MODEL.gguf SHARE_DIR OUT_PREFIX DOC.ids...\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1], share = argv[2], out_prefix = argv[3];
    const int K = std::atoi(std::getenv("PROBE_K") ? std::getenv("PROBE_K") : "128");
    std::vector<int> Rs = parse_ints("PROBE_R", "128,256");
    std::sort(Rs.begin(), Rs.end());
    const std::string arms = std::getenv("PROBE_ARMS") ? std::getenv("PROBE_ARMS") : "ABCD";
    const auto want = [&](char a) { return arms.find(a) != std::string::npos; };

    std::vector<std::string> docs;
    std::vector<std::vector<int32_t>> ids;
    size_t max_n = 0;
    for (int i = 4; i < argc; ++i) {
        std::vector<int32_t> d = read_ids(argv[i]);
        if ((int) d.size() < 1 + 256 + K || d[0] != 0) {
            std::fprintf(stderr, "FAIL: %s is not [BOS, D, C] (n=%zu)\n", argv[i], d.size());
            return 2;
        }
        max_n = std::max(max_n, d.size());
        docs.push_back(std::filesystem::path(argv[i]).filename().string());
        ids.push_back(std::move(d));
    }

    // Rows already written (resume): "doc|R|arm".
    std::set<std::string> done;
    {
        std::ifstream in(out_prefix + ".jsonl");
        for (std::string line; std::getline(in, line);) {
            const auto field = [&](const char * key) {
                const size_t p = line.find(std::string("\"") + key + "\":");
                if (p == std::string::npos) return std::string();
                size_t b = p + std::strlen(key) + 3, e = line.find_first_of(",}", b);
                std::string v = line.substr(b, e - b);
                v.erase(std::remove(v.begin(), v.end(), '"'), v.end());
                return v;
            };
            done.insert(field("doc") + "|" + field("R") + "|" + field("arm"));
        }
    }
    std::ofstream jsonl(out_prefix + ".jsonl", std::ios::app);

    DeepSeek4Backend backend(make_config(model, (int) max_n + 64, share));
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init\n");
        return 1;
    }
    DaemonIO io;

    std::vector<Row> rows;
    const auto emit = [&](const Row & r) {
        rows.push_back(r);
        char buf[512];
        std::snprintf(buf, sizeof buf,
                      "{\"doc\":\"%s\",\"L\":%d,\"R\":%d,\"arm\":\"%c\",\"nll\":%.6f,\"kl\":%.6f,"
                      "\"top1\":%.6f,\"nll_a\":%.6f,\"wall_s\":%.2f}",
                      r.doc.c_str(), r.L, r.R, r.arm, r.nll, r.kl, r.top1, r.nll_a, r.wall_s);
        jsonl << buf << "\n";
        jsonl.flush();
        std::fprintf(stderr, "[probe] %s\n", buf);
    };

    for (size_t di = 0; di < docs.size(); ++di) {
        const std::vector<int32_t> & d = ids[di];
        const int L = (int) d.size() - 1 - K;
        const std::vector<int32_t> prompt(d.begin(), d.begin() + 1 + L);
        const std::vector<int32_t> truth(d.begin() + 1 + L, d.end());
        bool all_done = true;
        for (int R : Rs)
            for (char a : std::string("ABCD"))
                if (want(a) && !done.count(docs[di] + "|" + std::to_string(R) + "|" + a)) all_done = false;
        if (all_done) { std::fprintf(stderr, "[probe] %s: done, skipping\n", docs[di].c_str()); continue; }

        std::vector<int> starts;
        for (int R : Rs) starts.push_back(replay_start(L, R));
        std::fprintf(stderr, "[probe] doc %zu/%zu %s L=%d starts:", di + 1, docs.size(), docs[di].c_str(), L);
        for (int s : starts) std::fprintf(stderr, " %d", s);
        std::fprintf(stderr, "\n");

        const auto base = [&](std::vector<int32_t> p) {
            GenerateRequest r;
            r.prompt = std::move(p);
            r.restore_points = starts;
            return r;
        };

        // A: native chain through the checkpoints; slot i holds the snapshot at starts[i].
        Logits logits_a;
        {
            const double t0 = now_s();
            luce_test::ScopedEnvVar floor_off(kFloorEnv, "0");
            bool ok = true;
            for (size_t i = 0; i < starts.size() && ok; ++i) {
                GenerateRequest r = base(std::vector<int32_t>(prompt.begin(), prompt.begin() + starts[i]));
                r.snap_slot = (int) i;
                r.snap_pos = starts[i];
                const GenerateResult res = i == 0 ? backend.generate(r, io)
                                                  : backend.restore_and_generate((int) i - 1, r, io);
                ok = res.ok() && backend.snapshot_used((int) i);
            }
            ok = ok && teacher_force(backend, truth, logits_a, [&] {
                     GenerateRequest r = base(prompt);
                     return backend.restore_and_generate((int) starts.size() - 1, r, io);
                 });
            if (!ok) { std::fprintf(stderr, "FAIL: arm A on %s\n", docs[di].c_str()); return 1; }
            const ArmScore sc = score_arm(logits_a, logits_a, truth);
            for (int R : Rs) {
                if (!want('A') || done.count(docs[di] + "|" + std::to_string(R) + "|A")) continue;
                emit({docs[di], L, R, 'A', sc.nll, 0.0, 1.0, sc.nll, now_s() - t0});
            }
        }
        const double nll_a = score_arm(logits_a, logits_a, truth).nll;

        for (size_t ri = 0; ri < Rs.size(); ++ri) {
            const int R = Rs[ri], s = starts[ri], slot = (int) ri;
            const auto key = [&](char a) { return docs[di] + "|" + std::to_string(R) + "|" + a; };
            const auto score_and_emit = [&](char arm, const Logits & lg, double wall) {
                const ArmScore sc = score_arm(logits_a, lg, truth);
                emit({docs[di], L, R, arm, sc.nll, sc.kl, sc.top1, nll_a, wall});
            };

            if (want('B') && !done.count(key('B'))) {
                // The floor makes the snapshot's rings irrelevant; zero them to show it.
                for (ggml_tensor * t : snapshot_tensors(backend.snapshot_ref(slot).ctx, "ds4_snap_raw_kv_"))
                    zero_tensor(t);
                luce_test::ScopedEnvVar floor_on(kFloorEnv, "1");
                Logits lg;
                const double t0 = now_s();
                if (!teacher_force(backend, truth, lg, [&] {
                        return backend.restore_and_generate(slot, base(prompt), io);
                    })) return 1;
                score_and_emit('B', lg, now_s() - t0);
            }
            if (want('D') && !done.count(key('D'))) {
                ggml_context * ctx = backend.snapshot_ref(slot).ctx;
                for (ggml_tensor * t : snapshot_tensors(ctx, "ds4_snap_raw_kv_")) zero_tensor(t);
                for (ggml_tensor * t : snapshot_tensors(ctx, "ds4_snap_comp_kv_")) zero_tensor(t);
                for (ggml_tensor * t : snapshot_tensors(ctx, "ds4_snap_index_kv_")) zero_tensor(t);
                luce_test::ScopedEnvVar floor_on(kFloorEnv, "1");
                Logits lg;
                const double t0 = now_s();
                if (!teacher_force(backend, truth, lg, [&] {
                        return backend.restore_and_generate(slot, base(prompt), io);
                    })) return 1;
                score_and_emit('D', lg, now_s() - t0);
            }
            if (want('C') && !done.count(key('C'))) {
                luce_test::ScopedEnvVar floor_off(kFloorEnv, "0");
                std::vector<int32_t> p{prompt[0]};
                p.insert(p.end(), prompt.begin() + s, prompt.end());
                Logits lg;
                const double t0 = now_s();
                if (!teacher_force(backend, truth, lg, [&] {
                        GenerateRequest r;
                        r.prompt = p;
                        return backend.generate(r, io);
                    })) return 1;
                score_and_emit('C', lg, now_s() - t0);
            }
        }
        for (size_t i = 0; i < starts.size(); ++i) backend.snapshot_free((int) i);
    }

    // Summary over every row in the file (this run and earlier ones).
    {
        std::ifstream in(out_prefix + ".jsonl");
        rows.clear();
        for (std::string line; std::getline(in, line);) {
            Row r;
            char doc[256], arm;
            if (std::sscanf(line.c_str(),
                            "{\"doc\":\"%255[^\"]\",\"L\":%d,\"R\":%d,\"arm\":\"%c\",\"nll\":%lf,\"kl\":%lf,"
                            "\"top1\":%lf,\"nll_a\":%lf,\"wall_s\":%lf}",
                            doc, &r.L, &r.R, &arm, &r.nll, &r.kl, &r.top1, &r.nll_a, &r.wall_s) == 9) {
                r.doc = doc; r.arm = arm;
                rows.push_back(r);
            }
        }
        std::map<std::tuple<int, int, char>, std::vector<const Row *>> groups;
        for (const Row & r : rows) groups[{r.L, r.R, r.arm}].push_back(&r);
        std::ofstream md(out_prefix + ".summary.md");
        const char * hdr = "| L | R | arm | n | NLL_A med | dNLL med | dNLL mean | KL med | KL mean | top1 med | top1 mean | wall med (s) |\n"
                           "|---|---|-----|---|-----------|----------|-----------|--------|---------|----------|-----------|--------------|\n";
        md << hdr;
        std::fputs(hdr, stderr);
        for (const auto & [k, g] : groups) {
            std::vector<double> nll_a, dn, kl, t1, wall;
            for (const Row * r : g) {
                nll_a.push_back(r->nll_a); dn.push_back(r->nll - r->nll_a);
                kl.push_back(r->kl); t1.push_back(r->top1); wall.push_back(r->wall_s);
            }
            char buf[400];
            std::snprintf(buf, sizeof buf, "| %d | %d | %c | %zu | %.4f | %+.4f | %+.4f | %.4f | %.4f | %.3f | %.3f | %.1f |\n",
                          std::get<0>(k), std::get<1>(k), std::get<2>(k), g.size(), median(nll_a),
                          median(dn), mean(dn), median(kl), mean(kl), median(t1), mean(t1), median(wall));
            md << buf;
            std::fputs(buf, stderr);
        }
    }
    return 0;
}
