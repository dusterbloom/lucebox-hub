// CPU-only: c++ -std=c++17 -Iserver/src server/test/test_qwen4exp_mtp.cpp -o /tmp/test_qwen4exp_mtp
#include "qwen4exp/qwen4exp_mtp.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace luce::common;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (0)

int main() {
    CHECK(qwen4exp_mtp_draft_length(nullptr) == 1);
    for (const char * v : {"", "0", "-3", "garbage", "2x", "2.5", "999999999999999999999x"}) {
        CHECK(qwen4exp_mtp_draft_length(v) == 1);
    }
    CHECK(qwen4exp_mtp_draft_length("2") == 2);
    CHECK(qwen4exp_mtp_draft_length("3") == 3);
    CHECK(qwen4exp_mtp_draft_length("4") == 4);
    CHECK(qwen4exp_mtp_draft_length("5") == 4);
    CHECK(qwen4exp_mtp_draft_length("999999999999999999999") == 4);
    CHECK(qwen4exp_mtp_draft_length("-999999999999999999999") == 1);

    int cases = 0;
    for (int k = 0; k <= 4; ++k) {
        std::array<int32_t, 5> drafts{11, 22, 33, 44, 55}, samples{};
        // Every equality pattern, including later matches following a reject.
        for (int mask = 0; mask < (1 << (k + 1)); ++mask) {
            for (int i = 0; i <= k; ++i) samples[i] = drafts[i] + ((mask >> i) & 1);
            for (int n = 0; n <= k + 1; ++n) {
                const auto result = qwen4exp_mtp_accept(drafts.data(), k, samples.data(), n);
                int accepted = 0;
                while (accepted < k && accepted < n && samples[accepted] == drafts[accepted]) ++accepted;
                const int emitted = std::min(n, accepted + 1);
                CHECK(result.n_accepted == accepted);
                CHECK(result.n_emitted == emitted);
                for (int i = 0; i < emitted; ++i) CHECK(result.emitted[i] == samples[i]);
                ++cases;
            }
        }
    }

    // Rollback at all block alignments and accepted prefixes, including a
    // five-token verify that completes TWO blocks. Stale pooled rows must be
    // excluded, then recomputed from replacement raw keys on block completion.
    for (int k = 1; k <= 4; ++k) {
        for (int alignment = 0; alignment < 4; ++alignment) {
            for (int retained = 1; retained <= k + 1; ++retained) {
                const int pos = 2200 + alignment, end = pos + retained;
                const int verified_blocks = (pos + k + 1) / 4;
                const int blocks = qwen4exp_mtp_retained_blocks(verified_blocks, end, 4);
                CHECK(blocks == end / 4);
                CHECK(qwen4exp_mtp_retained_blocks(0, end, 4) == 0); // dense prefix has no pooled keys
                std::vector<int> raw(pos + k + 8), pooled(raw.size() / 4, -1);
                for (size_t i = 0; i < raw.size(); ++i) raw[i] = (int) i;
                auto pool = [&](int block) {
                    return raw[4 * block] + raw[4 * block + 1] + raw[4 * block + 2] + raw[4 * block + 3];
                };
                for (int b = 0; b < verified_blocks; ++b) pooled[b] = pool(b);
                const int next_end = (end / 4 + 1) * 4;
                for (int p = end; p < next_end; ++p) raw[p] = -p;
                for (int b = blocks; b < next_end / 4; ++b) pooled[b] = pool(b);
                for (int b = 0; b < next_end / 4; ++b) CHECK(pooled[b] == pool(b));
                ++cases;
            }
        }
    }
    std::printf("qwen4exp MTP: %d acceptance/rollback cases passed; env validation passed\n", cases);
}
