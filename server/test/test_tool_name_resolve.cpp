// Unit tests for tool_name_levenshtein() and resolve_tool_name().
// Pure logic; no GPU, no model files required.
//
// Build:  ninja -C /tmp/rynesn-on-main-build test_tool_name_resolve
// Run:    /tmp/rynesn-on-main-build/test_tool_name_resolve

#include "server/tool_parser.h"
#include "server/tool_parser_internal.h"

#include <cstdio>
#include <string>

using namespace dflash::common;

// ── Minimal test framework ────────────────────────────────────────────────

static int failures = 0;
static int count    = 0;

#define CHECK(cond, msg) do {                                                  \
    count++;                                                                   \
    if (cond) {                                                                \
        std::printf("PASS %s\n", (msg));                                       \
    } else {                                                                   \
        std::printf("FAIL %s\n", (msg));                                       \
        failures++;                                                            \
    }                                                                          \
} while (0)

#define CHECK_EQ(a, b, msg) do {                                               \
    count++;                                                                   \
    if ((a) == (b)) {                                                          \
        std::printf("PASS %s\n", (msg));                                       \
    } else {                                                                   \
        std::printf("FAIL %s  (got '%s', want '%s')\n",                        \
                    (msg), (a).c_str(), (b).c_str());                          \
        failures++;                                                            \
    }                                                                          \
} while (0)

// ── Tool fixture builders ─────────────────────────────────────────────────

// Shape 1: {"name":"Bash", ...}  (flat)
static json flat_tool(const std::string & name) {
    return json{{"name", name}};
}

// Shape 2: {"function":{"name":"Bash", ...}}  (nested)
static json nested_tool(const std::string & name) {
    return json{{"type", "function"}, {"function", {{"name", name}}}};
}

// Build a tools array mixing both shapes for robustness.
static json make_tools(const std::vector<std::string> & names) {
    json arr = json::array();
    for (size_t i = 0; i < names.size(); i++) {
        // Alternate shapes to verify both are accepted.
        if (i % 2 == 0) arr.push_back(flat_tool(names[i]));
        else             arr.push_back(nested_tool(names[i]));
    }
    return arr;
}

// ── Levenshtein tests ─────────────────────────────────────────────────────

static void test_levenshtein() {
    // "bash" vs "bsh" → 1 deletion
    CHECK(tool_name_levenshtein("bash", "bsh") == 1,
          "levenshtein(bash,bsh)==1");
    // identical
    CHECK(tool_name_levenshtein("Bash", "Bash") == 0,
          "levenshtein(Bash,Bash)==0");
    // completely different short strings
    CHECK(tool_name_levenshtein("Read", "Bash") == 4,
          "levenshtein(Read,Bash)==4");
    // empty
    CHECK(tool_name_levenshtein("", "") == 0,
          "levenshtein(empty,empty)==0");
    CHECK(tool_name_levenshtein("abc", "") == 3,
          "levenshtein(abc,empty)==3");
}

// ── resolve_tool_name tests ───────────────────────────────────────────────

static void test_resolve_synonyms() {
    // Both flat and nested tool shapes in the fixture.
    json tools = json::array({flat_tool("Bash"), nested_tool("Read")});

    // Synonym: Execute → Bash
    CHECK_EQ(resolve_tool_name(tools, "Execute"), std::string("Bash"),
             "resolve Execute->Bash (synonym)");

    // Synonym: Shell → Bash
    CHECK_EQ(resolve_tool_name(tools, "Shell"), std::string("Bash"),
             "resolve Shell->Bash (synonym)");

    // Synonym: run → Bash
    CHECK_EQ(resolve_tool_name(tools, "run"), std::string("Bash"),
             "resolve run->Bash (synonym)");
}

static void test_resolve_case_insensitive() {
    json tools = make_tools({"Bash", "Read"});

    // Case-insensitive exact match
    CHECK_EQ(resolve_tool_name(tools, "BASH"), std::string("Bash"),
             "resolve BASH->Bash (case-insensitive)");
}

static void test_resolve_exact_valid() {
    json tools = make_tools({"Bash", "Read"});

    // Already valid — must return byte-identical (no mutation)
    CHECK_EQ(resolve_tool_name(tools, "Bash"), std::string("Bash"),
             "resolve Bash->Bash (already valid, unchanged)");

    CHECK_EQ(resolve_tool_name(tools, "Read"), std::string("Read"),
             "resolve Read->Read (already valid, unchanged)");
}

static void test_resolve_unknown_no_synonym_rejected() {
    json tools = make_tools({"Bash", "Read"});

    // "WebSearch" has no synonym, edit-dist to Bash=7, to Read=7 → refuse
    CHECK_EQ(resolve_tool_name(tools, "WebSearch"), std::string(""),
             "resolve WebSearch->'' (no synonym, dist>2)");
}

static void test_resolve_synonym_target_absent() {
    // Synonym "Execute"→Bash, but Bash is NOT in tools → refuse, never invent
    json tools = make_tools({"Read"});
    CHECK_EQ(resolve_tool_name(tools, "Execute"), std::string(""),
             "resolve Execute->'' (synonym Bash absent, hard invariant)");
}

static void test_resolve_ambiguous_edit_distance() {
    // Both "Reads" and "Ready" are distance 1 from "Read" → ambiguous → refuse
    json tools = make_tools({"Reads", "Ready"});
    CHECK_EQ(resolve_tool_name(tools, "Read"), std::string(""),
             "resolve Read->'' (ambiguous dist-1 to Reads and Ready)");
}

static void test_resolve_exact_unusual_name() {
    // "Bath" is a valid tool (exact match wins)
    json tools = make_tools({"Bath"});
    CHECK_EQ(resolve_tool_name(tools, "Bath"), std::string("Bath"),
             "resolve Bath->Bath (exact valid wins)");
}

// ── End-to-end through parse_tool_calls ───────────────────────────────────

static void test_e2e_hallucinated_execute_no_leading_bracket() {
    // Exact observed malformed string from the bug report.
    // Note: "function=Execute>" has NO leading '<'.
    // The current 6 patterns all require '<function=' so this WOULD be missed
    // without a new parser extension. After the fix, it must yield 1 call
    // named "Bash".
    std::string text =
        "Now let's run the test suite again:\n\n"
        "function=Execute>\n"
        "<parameter=command>\n"
        "cd /x && pytest\n"
        "</parameter>\n"
        "</function>\n";

    json tools = json::array({flat_tool("Bash")});
    auto result = parse_tool_calls(text, tools);

    CHECK(result.tool_calls.size() == 1,
          "e2e: bracket-less function=Execute yields 1 tool call");
    if (!result.tool_calls.empty()) {
        CHECK_EQ(result.tool_calls[0].name, std::string("Bash"),
                 "e2e: bracket-less Execute corrected to Bash");
    }
}

// ── main ──────────────────────────────────────────────────────────────────

int main() {
    std::printf("=== test_tool_name_resolve ===\n");

    test_levenshtein();
    test_resolve_synonyms();
    test_resolve_case_insensitive();
    test_resolve_exact_valid();
    test_resolve_unknown_no_synonym_rejected();
    test_resolve_synonym_target_absent();
    test_resolve_ambiguous_edit_distance();
    test_resolve_exact_unusual_name();
    test_e2e_hallucinated_execute_no_leading_bracket();

    std::printf("\n%d/%d passed\n", count - failures, count);
    return failures == 0 ? 0 : 1;
}
