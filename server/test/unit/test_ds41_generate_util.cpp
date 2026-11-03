// Host-only checks for test/ds41_bridge_generate_util.h.
//   g++ -std=c++17 -I server/test server/test/unit/test_ds41_generate_util.cpp
#include "ds41_bridge_generate_util.h"

#include <array>
#include <cstdio>

using namespace ds41gen;

static int failures = 0;
static void check(bool ok, const char * what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

static void put32(std::vector<uint8_t> & b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}

// Layers: {layer, ratio, n_rows, width}; data is a counter so order is checkable.
static std::vector<uint8_t> make_rows(const std::vector<std::array<uint32_t, 4>> & layers) {
    std::vector<uint8_t> b = {'D', 'S', '4', '1', 'R', 'O', 'W', 'S'};
    put32(b, 1);
    put32(b, (uint32_t) layers.size());
    for (const auto & l : layers) for (uint32_t v : l) put32(b, v);
    uint16_t c = 1;
    for (const auto & l : layers)
        for (uint32_t i = 0; i < l[2] * l[3]; ++i, ++c) { b.push_back(c & 255); b.push_back(c >> 8); }
    return b;
}

int main() {
    // sha256 known answers.
    check(sha256_hex(nullptr, 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
    const char * abc = "abc";
    check(sha256_hex((const uint8_t *) abc, 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
    const std::string two_block(56, 'a');  // 56 bytes: padding spills into a second block
    check(sha256_hex((const uint8_t *) two_block.data(), two_block.size()) ==
          "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a", "sha256 56 bytes");
    const int32_t one = 1;
    const uint8_t le1[4] = {1, 0, 0, 0};
    check(sha256_ids(&one, 1) == sha256_hex(le1, 4), "sha256_ids is LE int32 bytes");

    // DS41ROWS round trip: two layers (comp ratio 4 width 3, index ratio 4 width 2), s = 16.
    {
        const auto buf = make_rows({{{2, 4, 4, 3}}, {{8, 4, 4, 2}}});
        RowsFile f; std::string err;
        check(parse_rows(buf.data(), buf.size(), f, err), "parse ok");
        check(f.layers.size() == 2 && f.find(8) && f.find(2) && !f.find(3), "layers found by id");
        const RowsLayer * l2 = f.find(2);
        check(l2 && l2->ratio == 4 && l2->n_rows == 4 && l2->width == 3 && l2->f16.size() == 12, "layer 2 header");
        check(l2 && l2->f16[0] == 1 && l2->f16[11] == 12, "layer 2 data order");
        check(f.find(8)->f16[0] == 13 && f.find(8)->f16[7] == 20, "layer 8 follows layer 2");
        check(validate_rows_layer(*l2, 16, 3, 4, err), "shape matches");
        check(!validate_rows_layer(*l2, 16, 4, 4, err) && err.find("width") != std::string::npos, "width mismatch");
        check(!validate_rows_layer(*l2, 16, 3, 5, err) && err.find("rows") != std::string::npos, "snapshot has more rows than file");
        check(!validate_rows_layer(*l2, 16, 3, 3, err), "snapshot has fewer rows than file");
        check(!validate_rows_layer(*l2, 20, 3, 4, err) && err.find("s/ratio") != std::string::npos, "rows != s/ratio");
    }
    // Bad inputs.
    {
        RowsFile f; std::string err;
        auto buf = make_rows({{{2, 4, 4, 3}}});
        auto bad = buf; bad[0] = 'X';
        check(!parse_rows(bad.data(), bad.size(), f, err) && err == "bad magic", "bad magic");
        bad = buf; bad[8] = 2;
        check(!parse_rows(bad.data(), bad.size(), f, err) && err == "unsupported version", "bad version");
        bad = buf; bad.pop_back();
        check(!parse_rows(bad.data(), bad.size(), f, err) && err == "truncated data", "truncated data");
        bad = buf; bad.push_back(0);
        check(!parse_rows(bad.data(), bad.size(), f, err) && err == "trailing bytes", "trailing bytes");
        check(!parse_rows(buf.data(), 10, f, err), "short header");
        check(!parse_rows(buf.data(), 20, f, err) && err == "truncated layer headers", "truncated layer headers");
        const auto dup = make_rows({{{2, 4, 1, 1}}, {{2, 4, 1, 1}}});
        check(!parse_rows(dup.data(), dup.size(), f, err) && err == "duplicate layer", "duplicate layer");
        const auto zr = make_rows({{{2, 0, 1, 1}}});
        check(!parse_rows(zr.data(), zr.size(), f, err), "zero ratio");
        const auto empty = make_rows({});
        check(parse_rows(empty.data(), empty.size(), f, err) && f.layers.empty(), "zero layers parses");
    }

    // Tool-block detection. open = [10,11,12], close = [20,10,11,12]-like: close starts with a
    // distinct token so open and close never alias.
    const std::vector<int32_t> open = {100, 7, 8}, close = {101, 7, 8};
    {
        const std::vector<int32_t> ids = {1, 2, 100, 7, 8, 5, 6, 101, 7, 8, 9, 9};
        const ToolSpan s = find_tool_block(ids, open, close);
        check(s.complete && s.begin == 2 && s.end == 10, "complete block span");
        ToolBlockTracker t(open, close);
        int stop_at = -1;
        for (size_t i = 0; i < ids.size(); ++i) if (t.push(ids[i])) { stop_at = (int) i; break; }
        check(stop_at == 9, "tracker fires on the last close-marker token");
    }
    {
        const ToolSpan s = find_tool_block({1, 100, 7, 8, 5, 6}, open, close);
        check(!s.complete && s.begin == 1 && s.end == -1, "open without close");
        const ToolSpan n = find_tool_block({1, 2, 3, 7, 8}, open, close);
        check(!n.complete && n.begin == -1, "no markers");
        const ToolSpan c = find_tool_block({1, 101, 7, 8, 100, 7, 8}, open, close);
        check(!c.complete && c.begin == 4, "close before open is ignored");
        const ToolSpan p = find_tool_block({100, 7, 8, 101, 7}, open, close);
        check(!p.complete, "partial close marker");
        const ToolSpan z = find_tool_block({100, 7, 8, 5, 101, 7, 8, 100, 7, 8, 101, 7, 8}, open, close);
        check(z.complete && z.begin == 0 && z.end == 7, "first block wins");
        check(count_marker({100, 7, 8, 1, 100, 7, 8}, open) == 2, "count_marker");
    }

    // Comparison.
    {
        const std::vector<int32_t> a = {1, 2, 100, 7, 8, 5, 6, 101, 7, 8};
        GenInfo ga{a, find_tool_block(a, open, close), "read_file"};
        check(ga.span.complete, "ga span");
        GenInfo same = ga;
        StepCompare c = compare_next_step(ga, same);
        check(c.same_step && c.same_tool && c.first_divergence == -1, "identical");
        // Different reasoning before the block, same block: same_step, divergence at 1.
        std::vector<int32_t> b = {1, 3, 4, 100, 7, 8, 5, 6, 101, 7, 8};
        GenInfo gb{b, find_tool_block(b, open, close), "read_file"};
        c = compare_next_step(ga, gb);
        check(c.same_step && c.same_tool && c.first_divergence == 1, "same block, different reasoning");
        // Same tool, different argument.
        std::vector<int32_t> d = {1, 2, 100, 7, 8, 5, 9, 101, 7, 8};
        GenInfo gd{d, find_tool_block(d, open, close), "read_file"};
        c = compare_next_step(ga, gd);
        check(!c.same_step && c.same_tool && c.first_divergence == 6, "same tool, different args");
        gd.tool_name = "bash";
        check(!compare_next_step(ga, gd).same_tool, "different tool");
        // Arm never closes a block (cap).
        GenInfo cap{{1, 2, 100, 7, 8, 5}, find_tool_block({1, 2, 100, 7, 8, 5}, open, close), ""};
        c = compare_next_step(ga, cap);
        check(!c.same_step && !c.same_tool && c.first_divergence == 6, "arm without block; prefix divergence = shorter length");
        // Reference without a block never counts as same.
        c = compare_next_step(cap, ga);
        check(!c.same_step && !c.same_tool, "reference without block");
        check(first_divergence({}, {}) == -1 && first_divergence({}, {4}) == 0, "empty divergence");
        // Stops.
        check(classify_stop(ga, {50}) == StopKind::Tool, "stop tool");
        check(classify_stop(GenInfo{{1, 50}, {}, ""}, {50, 51}) == StopKind::Eos, "stop eos");
        check(classify_stop(cap, {50}) == StopKind::Cap, "stop cap");
        check(std::string(stop_name(StopKind::Cap)) == "cap", "stop name");
    }

    // Tool-name extraction on synthetic DSML (V4.1 leading-space tags, and V4).
    {
        const std::string v41 =
            "\n\n<\xEF\xBD\x9C" "DSML\xEF\xBD\x9C calls>\n<\xEF\xBD\x9C" "DSML\xEF\xBD\x9C invoke name=\"read_file\">\n"
            "<\xEF\xBD\x9C" "DSML\xEF\xBD\x9C parameter name=\"path\" string=\"true\">a.txt</\xEF\xBD\x9C" "DSML\xEF\xBD\x9C parameter>\n"
            "</\xEF\xBD\x9C" "DSML\xEF\xBD\x9C invoke>\n</\xEF\xBD\x9C" "DSML\xEF\xBD\x9C calls>";
        check(extract_tool_name(v41) == "read_file", "v4.1 name");
        const std::string v4 =
            "<\xEF\xBD\x9C" "DSML\xEF\xBD\x9Ctool_calls>\n<\xEF\xBD\x9C" "DSML\xEF\xBD\x9Cinvoke name=\"bash\">x</\xEF\xBD\x9C"
            "DSML\xEF\xBD\x9Cinvoke>";
        check(extract_tool_name(v4) == "bash", "v4 name");
        check(extract_tool_name("<x invoke name=\"a.b-c\">\" invoke name=\"second\"") == "a.b-c", "first invoke wins");
        check(extract_tool_name("no block here").empty(), "no invoke");
        check(extract_tool_name("invoke name=\"unterminated").empty(), "unterminated name");
        check(extract_tool_name("</x invoke>").empty(), "closing tag has no name");
    }

    // Per-step scoring: logits (0, ln 3): token 1 has p = 3/4.
    {
        const float lg[2] = {0.0f, (float) std::log(3.0)};
        double nll = 0; bool top1 = false;
        score_step(lg, 2, 1, nll, top1);
        check(std::fabs(nll - std::log(4.0 / 3.0)) < 1e-6 && top1, "score_step likelier token");
        score_step(lg, 2, 0, nll, top1);
        check(std::fabs(nll - std::log(4.0)) < 1e-6 && !top1, "score_step other token");
        const float tie[3] = {1.0f, 1.0f, 0.0f};
        score_step(tie, 3, 1, nll, top1);
        check(!top1, "score_step tie -> first index");
    }

    // JSON helpers.
    {
        const std::string row = "{\"key\":\"abc\",\"arm\":\"Bt\",\"s\":8062,\"nll\":0.5,\"first_divergence\":-1,\"x\":null}";
        check(json_field(row, "key") == "abc" && json_field(row, "arm") == "Bt", "json_field strings");
        check(json_field(row, "x") == "null" && json_field(row, "nope").empty(), "json_field null/missing");
        int64_t v = 0;
        check(json_int(row, "s", v) && v == 8062 && json_int(row, "first_divergence", v) && v == -1, "json_int");
        check(!json_int(row, "key", v) && !json_int(row, "nope", v), "json_int rejects non-numbers");
        const std::string meta = "{\n \"s\": 8062,\n \"k_start\": 8190,\n \"k_len\": 128,\n \"D_len\": 8000\n}";
        check(json_int(meta, "k_start", v) && v == 8190 && json_int(meta, "D_len", v) && v == 8000, "meta fields");
        check(json_num(NAN) == "null" && json_num(0.5) == "0.500000", "json_num");
    }

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_ds41_generate_util: all checks passed\n");
    return 0;
}
