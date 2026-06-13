// DiffusionBackend implementation. See diffusion_backend.h.

#include "diffusion_backend.h"
#include "diffusion_decoder.h"

#include <chrono>
#include <cstdio>
#include <utility>

namespace dflash::common {

DiffusionBackend::DiffusionBackend(std::unique_ptr<DiffusionModelGraph> model,
                                   DiffusionConfig cfg,
                                   std::string arch_label)
    : model_(std::move(model)), cfg_(cfg), arch_label_(std::move(arch_label)) {}

DiffusionBackend::~DiffusionBackend() = default;

void DiffusionBackend::print_ready_banner() const {
    const char * scheme = (cfg_.noise_scheme == DiffusionNoise::UniformState)
                              ? "uniform-state" : "masked";
    std::printf("[%s-diffusion-daemon] ready (block=%d steps=%d noise=%s)\n",
                arch_label_.c_str(), cfg_.block_size,
                cfg_.n_steps > 0 ? cfg_.n_steps : cfg_.block_size, scheme);
}

bool DiffusionBackend::park(const std::string & /*what*/) { return true; }
bool DiffusionBackend::unpark(const std::string & /*what*/) { return true; }

GenerateResult DiffusionBackend::generate_impl(const GenerateRequest & req,
                                               const DaemonIO & io) {
    GenerateResult r;
    if (!model_) { r.error = "no_model"; return r; }

    // Adapt the server's DaemonIO (stream fd + client-disconnect callback) to
    // the decoder's plain token sink. io.emit() writes to the stream fd and
    // invokes io.on_token, flipping io.cancelled when the client disconnects.
    DiffusionStream stream;
    stream.on_token = [&io](int32_t tok) -> bool {
        io.emit(tok);
        return !io.cancelled;
    };

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    DiffusionDecodeResult d = run_diffusion_generate(
        *model_, req.prompt, req.n_gen, cfg_, req.sampler, req.do_sample, stream);
    const auto t1 = clock::now();

    r.ok        = d.ok;
    r.error     = d.error;
    r.tokens    = std::move(d.tokens);
    r.prefill_s = 0.0;  // prepare() time is folded into decode for now
    r.decode_s  = std::chrono::duration<double>(t1 - t0).count();
    return r;
}

// ── Unsupported AR-only surface ──────────────────────────────────────────
bool DiffusionBackend::snapshot_save(int /*slot*/)        { return false; }
void DiffusionBackend::snapshot_free(int /*slot*/)        {}
bool DiffusionBackend::snapshot_used(int /*slot*/) const  { return false; }
int  DiffusionBackend::snapshot_cur_pos(int /*slot*/) const { return -1; }

GenerateResult DiffusionBackend::restore_and_generate_impl(
        int /*slot*/, const GenerateRequest & /*req*/, const DaemonIO & /*io*/) {
    GenerateResult r;
    r.error = "restore_unsupported";  // diffusion has no causal KV snapshot to restore
    return r;
}

bool DiffusionBackend::handle_compress(const std::string & /*line*/,
                                       const DaemonIO & io) {
    std::fprintf(stderr, "[diffusion] pflash compress is not supported\n");
    io.emit(-1);
    return false;
}

void DiffusionBackend::shutdown() {
    if (model_) model_->reset();
    model_.reset();
}

}  // namespace dflash::common
