// A1 render parity: the server's own tokenizer (server/src/server/tokenizer.cpp, compiled unchanged) as a CPU command-line
// program. argv[1] = GGUF path (metadata only is read; a tokenizer-only GGUF works). stdin: one JSON object per line,
// {"text": "..."} -> {"ids": [...]}   or   {"decode": [ids]} -> {"text": "..."}.
#include "tokenizer.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

using nlohmann::json;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: tokenize_cli <model.gguf>\n";
        return 2;
    }
    luce::common::Tokenizer tok;
    if (!tok.load_from_gguf(argv[1])) return 1;
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json out;
        try {
            const json in = json::parse(line);
            if (in.contains("decode")) {
                const auto ids = in["decode"].get<std::vector<int32_t>>();
                out = json{{"text", tok.decode(ids)}};
            } else {
                out = json{{"ids", tok.encode(in.value("text", std::string()))}};
            }
        } catch (const std::exception & e) {
            out = json{{"error", e.what()}};
        }
        std::cout << out.dump(-1, ' ', false, json::error_handler_t::replace) << "\n" << std::flush;
    }
    return 0;
}
