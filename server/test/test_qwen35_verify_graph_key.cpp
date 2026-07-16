#include "qwen35/verify_graph_key.h"

#include <map>

using dflash::common::Qwen35VerifyGraphKey;

int main() {
    std::map<Qwen35VerifyGraphKey, int> graphs;
    graphs[{5, true, true}] = 1;
    graphs[{5, false, true}] = 2;
    graphs[{5, true, false}] = 3;
    graphs[{4, true, true}] = 4;
    if (graphs.size() != 4) return 1;
    if (graphs[{5, true, true}] != 1) return 2;
    if (graphs[{5, false, true}] != 2) return 3;
    if (graphs[{5, true, false}] != 3) return 4;
    if (graphs[{4, true, true}] != 4) return 5;
    return 0;
}
