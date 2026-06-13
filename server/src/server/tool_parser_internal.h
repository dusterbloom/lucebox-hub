// Internal helpers exposed for unit testing only.
// Production code uses tool_parser.h; do not include this header outside tests.
#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace dflash::common {

using json = nlohmann::json;

// Standard DP Levenshtein distance.
size_t tool_name_levenshtein(const std::string & a, const std::string & b);

// Map a (potentially hallucinated) tool name to the nearest valid tool name
// present in `tools`. Returns the corrected name, or "" if no confident
// mapping exists.  HARD INVARIANT: the returned string is always either ""
// or a name that already appears in `tools`.
std::string resolve_tool_name(const json & tools, const std::string & name);

}  // namespace dflash::common
