// A1 tool-parser program: reads one JSON object per line on stdin, {"text": "<generated text>", "tools": [<OpenAI tools>]},
// calls luce::common::parse_tool_calls from the server's tool_parser.cpp (compiled unchanged) and prints one JSON object per
// line: {"cleaned_text": "...", "tool_calls": [{"id","name","arguments"}]}. A line that is not valid JSON yields {"error": "..."}.
#include "tool_parser.h"

#include <iostream>
#include <string>

using luce::common::json;

static std::string dump(const json & j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json out;
        try {
            const json in = json::parse(line);
            const std::string text = in.value("text", std::string());
            const json tools = in.contains("tools") ? in["tools"] : json();
            const auto parsed = luce::common::parse_tool_calls(text, tools);
            out["cleaned_text"] = parsed.cleaned_text;
            out["tool_calls"] = json::array();
            for (const auto & c : parsed.tool_calls)
                out["tool_calls"].push_back({{"id", c.id}, {"name", c.name}, {"arguments", c.arguments}});
        } catch (const std::exception & e) {
            out = json{{"error", e.what()}};
        }
        std::cout << dump(out) << "\n";
    }
    return 0;
}
