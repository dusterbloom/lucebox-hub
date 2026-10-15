// Unit test for the KVFLASH_QK_DUMP line writer (pure; no GPU).
//   1. the dumped scores are exactly kvflash_qk_chunk_scores on the same data
//      (last-token query, and the max over per-token queries)
//   2. missing pooled keys dump as the -2 sentinel, residency is listed
//   3. env var unset: no directory, no file, append is a no-op
//   4. env var set: one JSON line per call is appended

#include "CppUnitTestFramework.hpp"

#define KVFLASH_QK_PURE_ONLY
#include "kvflash_qk.h"
#include "kvflash_qk_dump.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashQkDumpFixture {};

// Pull the float array "key":[a,b,c] out of a dump line.
std::vector<float> parse_array(const std::string & line, const std::string & key) {
    std::vector<float> v;
    const std::string pat = "\"" + key + "\":[";
    size_t p = line.find(pat);
    if (p == std::string::npos) return v;
    p += pat.size();
    const size_t e = line.find(']', p);
    std::stringstream ss(line.substr(p, e - p));
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::strtof(tok.c_str(), nullptr));
    return v;
}
}

TEST_CASE(KvflashQkDumpFixture, kvflash_qk_dump_suite) {
    KvFlashQkDims d;
    d.n_layers = 2; d.n_q_heads = 4; d.n_kv_heads = 2; d.head_dim = 8;
    const size_t kstride = (size_t)d.n_layers * d.n_kv_heads * d.head_dim;
    const size_t qsize   = (size_t)d.n_layers * d.n_q_heads * d.head_dim;

    // Deterministic pseudo-random data. Pooled keys are L2-normalized per (layer, head).
    uint32_t s = 12345u;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f; };
    const int n_chunks = 5;
    std::vector<std::vector<float>> keys((size_t)n_chunks, std::vector<float>(kstride));
    for (auto & k : keys) {
        for (float & x : k) x = rnd();
        for (int lh = 0; lh < d.n_layers * d.n_kv_heads; lh++) {
            float * h = k.data() + (size_t)lh * d.head_dim;
            double ss = 0; for (int i = 0; i < d.head_dim; i++) ss += (double)h[i] * h[i];
            const float inv = 1.0f / (float)std::sqrt(ss);
            for (int i = 0; i < d.head_dim; i++) h[i] *= inv;
        }
    }
    std::vector<const float *> pk((size_t)n_chunks, nullptr);
    for (int c = 0; c < n_chunks; c++) if (c != 3) pk[(size_t)c] = keys[(size_t)c].data();   // chunk 3 missing

    const int n_q = 3;
    std::vector<float> q_all((size_t)n_q * qsize);
    for (float & x : q_all) x = rnd();
    const float * q_last = q_all.data() + (size_t)(n_q - 1) * qsize;

    // Reference: the library scorer.
    std::vector<float> ref_last, ref_max, tmp;
    kvflash_qk_chunk_scores(pk, q_last, d, ref_last);
    for (int t = 0; t < n_q; t++) {
        kvflash_qk_chunk_scores(pk, q_all.data() + (size_t)t * qsize, d, tmp);
        if (t == 0) ref_max = tmp;
        else for (size_t c = 0; c < tmp.size(); c++) ref_max[c] = std::max(ref_max[c], tmp[c]);
    }

    KvFlashQkDumpMeta m;
    m.seq = 7; m.prompt_hash = 0xdeadbeefcafef00dULL; m.prompt_len = 500; m.kv_offset = 400;
    m.committed = 500; m.chunk_tokens = 64;
    const std::vector<uint8_t> resident{1, 0, 1, 0, 1};
        // What the engine does: score the last token, fold the per-token queries (in two batches) into a running max.
    std::vector<float> dump_last, dump_max;
    kvflash_qk_chunk_scores(pk, q_last, d, dump_last);
    kvflash_qk_fold_max(pk, q_all.data(), 2, d, dump_max);
    kvflash_qk_fold_max(pk, q_all.data() + 2 * qsize, 1, d, dump_max);
    const std::string line = kvflash_qk_dump_line(m, 4, dump_last, dump_max, n_q, resident);

    CHECK(line.find('\n') == std::string::npos);
    CHECK(line.find("\"n_chunks\":5") != std::string::npos);
    CHECK(line.find("\"n_pooled\":4") != std::string::npos);
    CHECK(line.find("\"n_q_tokens\":3") != std::string::npos);
    CHECK(line.find("\"prompt_len\":500") != std::string::npos);
    CHECK(line.find("\"prompt_hash\":\"deadbeefcafef00d\"") != std::string::npos);
    CHECK(line.find("\"resident\":[0,2,4]") != std::string::npos);

    const std::vector<float> got_last = parse_array(line, "scores_last");
    const std::vector<float> got_max  = parse_array(line, "scores_max");
    CHECK(got_last.size() == (size_t)n_chunks);
    CHECK(got_max.size() == (size_t)n_chunks);
    bool same_last = got_last.size() == ref_last.size();
    bool same_max  = got_max.size() == ref_max.size();
    for (size_t c = 0; same_last && c < ref_last.size(); c++) same_last = got_last[c] == ref_last[c];   // exact: %.9g round-trips
    for (size_t c = 0; same_max && c < ref_max.size(); c++)  same_max  = got_max[c]  == ref_max[c];
    CHECK(same_last);
    CHECK(same_max);
    CHECK(got_last[3] == -2.0f);      // missing sentinel
    CHECK(got_max[3] == -2.0f);

    // No per-token queries: scores_max is empty, the line still has scores_last.
    const std::string line0 = kvflash_qk_dump_line(m, 4, dump_last, std::vector<float>(), 0, resident);
    CHECK(parse_array(line0, "scores_max").empty());
    CHECK(parse_array(line0, "scores_last").size() == (size_t)n_chunks);
    CHECK(line0.find("\"n_q_tokens\":0") != std::string::npos);

    // Env var unset: no dir, append is a no-op and creates nothing.
    ::unsetenv("KVFLASH_QK_DUMP");
    CHECK(kvflash_qk_dump_dir().empty());
    namespace fs = std::filesystem;
    const fs::path dir = fs::current_path() / "kvflash_qk_dump_test_out";
    fs::remove_all(dir);
    CHECK(!kvflash_qk_dump_append("", line));
    CHECK(!fs::exists(dir));

    // Env var set: appends one line per call, creating the directory.
    ::setenv("KVFLASH_QK_DUMP", dir.c_str(), 1);
    CHECK(kvflash_qk_dump_dir() == dir.string());
    CHECK(kvflash_qk_dump_append(kvflash_qk_dump_dir(), line));
    CHECK(kvflash_qk_dump_append(kvflash_qk_dump_dir(), line0));
    {
        std::ifstream f(dir / "qk_dump.jsonl");
        std::string a, b, c;
        CHECK((bool)std::getline(f, a));
        CHECK((bool)std::getline(f, b));
        CHECK(!std::getline(f, c));
        CHECK(a == line);
        CHECK(b == line0);
    }
    ::unsetenv("KVFLASH_QK_DUMP");
    fs::remove_all(dir);

    // Folding more chunks later keeps the earlier maxima and extends with the sentinel.
    {
        std::vector<const float *> pk2(pk.begin(), pk.begin() + 2);
        std::vector<float> run;
        kvflash_qk_fold_max(pk2, q_all.data(), 1, d, run);
        CHECK(run.size() == 2);
        kvflash_qk_fold_max(pk, q_all.data() + qsize, 1, d, run);
        CHECK(run.size() == (size_t)n_chunks);
        std::vector<float> a, b;
        kvflash_qk_chunk_scores(pk, q_all.data() + qsize, d, b);
        CHECK(run[4] == b[4]);                       // chunk 4 only seen by the second query
        kvflash_qk_chunk_scores(pk2, q_all.data(), d, a);
        CHECK(run[0] == std::max(a[0], b[0]));
    }

    // Many queries (threaded fold) equal the sequential max.
    {
        const int nq = 37;
        std::vector<float> qs((size_t)nq * qsize);
        for (float & x : qs) x = rnd();
        std::vector<float> seq, par, t1;
        for (int t = 0; t < nq; t++) {
            kvflash_qk_chunk_scores(pk, qs.data() + (size_t)t * qsize, d, t1);
            if (t == 0) seq = t1;
            else for (size_t c = 0; c < t1.size(); c++) seq[c] = std::max(seq[c], t1[c]);
        }
        kvflash_qk_fold_max(pk, qs.data(), nq, d, par);
        CHECK(par == seq);
    }

    // Prompt hash: stable, order-sensitive.
    const std::vector<int32_t> t1{1, 2, 3}, t2{3, 2, 1};
    CHECK(kvflash_qk_prompt_hash(t1) == kvflash_qk_prompt_hash(t1));
    CHECK(kvflash_qk_prompt_hash(t1) != kvflash_qk_prompt_hash(t2));

    std::printf("ALL PASS\n");
}
