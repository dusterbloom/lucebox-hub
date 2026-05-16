// test_mtp_interface_contract.cpp — Interface contract for IMtpModule.
//
// Verifies that any IMtpModule (External or Native flavor) honors the
// LSP-safe lifecycle: flavor() is stable, hidden_size() is positive,
// attach() returns sensibly, reset_chain()/shutdown() are idempotent.
// Concrete impls (Gemma4MtpModule, Qwen36MtpModule) must pass this same
// suite under their own test targets in later PRs.

#include "common/mtp_interface.h"
#include "common/dflash_target.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace dflash27b;
using namespace dflash27b::mtp;

#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        std::fprintf(stderr, "%s:%d CHECK(%s) FAILED\n",                     \
                     __FILE__, __LINE__, #cond);                             \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

namespace {

// ── Minimal DFlashTarget stub ───────────────────────────────────────────
// Just enough for IMtpModule::attach() to bind against. Methods that
// the contract test does not exercise return harmless defaults.
struct StubTarget : DFlashTarget {
    int hidden = 64;
    bool verify_batch(const std::vector<int32_t> &, int, int &,
                      std::vector<int32_t> *) override { return true; }
    bool snapshot_kv() override { return true; }
    bool restore_kv() override { return true; }
    bool is_eos(int) const override { return false; }
    bool embed_tokens(const int32_t *, int, float *) const override { return true; }
    bool project_hidden_to_tokens(const float *, int,
                                  std::vector<int32_t> &) override { return true; }
    int hidden_size() const override { return hidden; }
    int mask_token_id() const override { return -1; }
    const std::vector<int> & capture_layer_ids() const override {
        static const std::vector<int> ids;
        return ids;
    }
};

// ── Fake ExternalDrafter ───────────────────────────────────────────────
struct FakeExternalMtp : IExternalDrafterMtp {
    int H = 64;
    int gamma_max_ = 4;
    bool attached_ = false;
    std::vector<int> donors_{0};
    int max_gamma() const override { return gamma_max_; }
    int hidden_size() const override { return H; }
    bool attach(DFlashTarget * t) override {
        attached_ = (t != nullptr && t->hidden_size() == H);
        return attached_;
    }
    void reset_chain() override {}
    void shutdown() override { attached_ = false; }
    bool step(const StepInput & in, StepOutput & out) override {
        out.draft_token = in.current_token + 1;  // deterministic
        out.draft_logit = 1.0f;
        out.next_hidden.assign(H, 0.0f);
        return true;
    }
    const std::vector<int> & donor_layers() const override { return donors_; }
    bool enable_target_hidden_capture(bool, int) override { return true; }
    void set_capture_row(int) override {}
    bool consume_captured_hidden(float * out, int dim) override {
        if (dim != H) return false;
        for (int i = 0; i < dim; i++) out[i] = 0.0f;
        return true;
    }
};

// ── Fake NativeHeads ───────────────────────────────────────────────────
struct FakeNativeMtp : INativeMtp {
    int H = 64;
    int n_heads_ = 2;
    bool attached_ = false;
    int max_gamma() const override { return n_heads_; }
    int hidden_size() const override { return H; }
    bool attach(DFlashTarget * t) override {
        attached_ = (t != nullptr && t->hidden_size() == H);
        return attached_;
    }
    void reset_chain() override {}
    void shutdown() override { attached_ = false; }
    int num_heads() const override { return n_heads_; }
    bool step_batch(int32_t cur, int /*base_pos*/,
                    std::vector<StepOutput> & out) override {
        out.clear();
        for (int i = 0; i < n_heads_; i++) {
            StepOutput s;
            s.draft_token = cur + 1 + i;
            s.draft_logit = 1.0f;
            out.push_back(std::move(s));
        }
        return true;
    }
};

template <typename Mtp>
void exercise_contract(Mtp & mtp, MtpFlavor expected_flavor) {
    StubTarget tgt;
    tgt.hidden = mtp.hidden_size();

    // flavor() is stable and matches the expected mixin.
    CHECK(mtp.flavor() == expected_flavor);
    CHECK(mtp.flavor() == expected_flavor);  // call twice — must not flip

    // hidden_size() positive, max_gamma() at least 1.
    CHECK(mtp.hidden_size() > 0);
    CHECK(mtp.max_gamma() >= 1);

    // attach with a compatible target succeeds.
    CHECK(mtp.attach(&tgt));

    // reset_chain() + shutdown() are idempotent (no-op on repeat).
    mtp.reset_chain();
    mtp.reset_chain();
    mtp.shutdown();
    mtp.shutdown();
}

}  // namespace

int main() {
    {
        FakeExternalMtp m;
        exercise_contract(m, MtpFlavor::ExternalDrafter);
        std::printf("[contract] ExternalDrafter OK\n");
    }
    {
        FakeNativeMtp m;
        exercise_contract(m, MtpFlavor::NativeHeads);
        std::printf("[contract] NativeHeads OK\n");
    }
    std::printf("[contract] all PASS\n");
    return 0;
}
