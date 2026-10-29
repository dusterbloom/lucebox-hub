// Encoder SWA bounded replay on snapshot restore (LUCE_DS41_ENCODER_BOUNDED_REPLAY=1).
//
// DS4.1's persistent prefix cache keeps only global KV (compressed + index
// rows); the sliding-window ring is not persisted, and a hit replays the last
// n_swa tokens with the raw window truncated at the replay start. Lucebox
// adopts the ring from the snapshot instead. With the opt-in on, the suffix
// prefill after a restore at s floors every layer's raw window at s, so the
// snapshot's ring contents no longer matter.
//
//   test_ds41_encoder_replay MODEL.gguf IDS.bin SHARE_DIR [GPU-profile-env...]
//
// IDS.bin: int32 token ids (HF tokenizer, BOS first), at least 600 of them.
// Model-gated (lucebox6, ds41-lucebox profile); not registered in ordinary CI.
//
//   T1 ring independence: floor on, continuation logits are bit-identical for
//      native / all-zero / random snapshot rings.
//   T2 parity: restoring the snapshot gives the logits of continuing the live
//      cache in place with the same floor, and the floor does change the
//      logits (it is not a no-op).
//   T3 regression: floor off, restore + suffix equals the cold prefill.
//   T4 multi-chunk: T1 again with restore points inside the suffix, so the
//      replay prefill runs several chunks / a layer-major pass of several bands.
#include "ds41_bridge_common.h"
#include "scoped_env.h"

using namespace ds41;

namespace {

constexpr const char * kFloorEnv = "LUCE_DS41_ENCODER_BOUNDED_REPLAY";
constexpr int kForced = 8;     // teacher-forced decode steps compared
constexpr int kReplay = 192;   // suffix tokens after the snapshot (>= n_swa = 128)

int g_failures = 0;

void report(const char * name, bool ok, const std::string & detail) {
    std::fprintf(stderr, "%s: %s  %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf IDS.bin SHARE_DIR\n", argv[0]);
        return 2;
    }
    const std::vector<int32_t> ids = read_ids(argv[2]);
    if (ids.size() < 600) {
        std::fprintf(stderr, "FAIL: need >= 600 ids in %s, got %zu\n", argv[2], ids.size());
        return 2;
    }
    // prompt = ids[:n], forced continuation = ids[n:n+kForced], snapshot at s.
    const int n = (int) ids.size() - kForced;
    const int s = ((n - kReplay) / 2) * 2;  // even: no ratio-2 pending compressor state
    const std::vector<int32_t> prompt(ids.begin(), ids.begin() + n);
    const std::vector<int32_t> prefix(ids.begin(), ids.begin() + s);
    const std::vector<int32_t> forced(ids.begin() + n, ids.end());

    DeepSeek4Backend backend(make_config(argv[1], n + kForced + 256, argv[3]));
    if (!backend.init()) {
        std::fprintf(stderr, "FAIL: backend init\n");
        return 1;
    }
    std::fprintf(stderr, "[test] n=%d s=%d suffix=%d forced=%d\n", n, s, n - s, kForced);

    DaemonIO io;
    const auto cold_prefix = [&] {  // live cache + slot 0 at s
        GenerateRequest r;
        r.prompt = prefix;
        r.n_gen = 0;
        r.snap_slot = 0;
        r.snap_pos = s;
        return backend.generate(r, io);
    };
    const auto live_prefix = [&] {  // live cache at s, snapshot left alone
        GenerateRequest r;
        r.prompt = prefix;
        r.n_gen = 0;
        return backend.generate(r, io);
    };
    const auto request = [&] {
        GenerateRequest r;
        r.prompt = prompt;
        r.n_gen = kForced;
        return r;
    };

    // Native ring bytes of the snapshot, to put back between modes.
    if (!cold_prefix().ok() || !backend.snapshot_used(0)) {
        std::fprintf(stderr, "FAIL: prefix prefill / snapshot at %d\n", s);
        return 1;
    }
    ggml_context * sctx = backend.snapshot_ref(0).ctx;
    if (!sctx) { std::fprintf(stderr, "FAIL: snapshot_ref\n"); return 1; }
    const std::vector<ggml_tensor *> rings = snapshot_tensors(sctx, "ds4_snap_raw_kv_");
    std::vector<std::vector<uint8_t>> native;
    for (ggml_tensor * t : rings) native.push_back(tensor_bytes(t));
    std::fprintf(stderr, "[test] %zu ring tensors in the snapshot\n", rings.size());
    const auto set_rings = [&](const char * mode) {
        for (size_t i = 0; i < rings.size(); ++i) {
            if (!std::strcmp(mode, "native")) set_tensor_bytes(rings[i], native[i]);
            else if (!std::strcmp(mode, "zero")) zero_tensor(rings[i]);
            else if (!randomize_tensor(rings[i], 1000 + i)) {
                std::fprintf(stderr, "FAIL: ring type %s not randomizable\n",
                             ggml_type_name(rings[i]->type));
                std::exit(1);
            }
        }
    };
    const auto restored = [&](Logits & out) {
        return teacher_force(backend, forced, out,
                             [&] { return backend.restore_and_generate(0, request(), io); });
    };

    // Warm-up: the first requests fill the streamed-expert cache, and a routing
    // near-tie flipping on cache state moves logits by O(1); measure only a
    // warm cache (T0 checks it is steady).
    {
        luce_test::ScopedEnvVar floor_off(kFloorEnv, "0");
        Logits warm;
        set_rings("native");
        for (int i = 0; i < 2; ++i) restored(warm);
    }

    // ── T0: the harness itself is deterministic (floor off, same restore twice)
    //        and in-place continuing equals restoring (floor off).
    {
        luce_test::ScopedEnvVar floor_off(kFloorEnv, "0");
        Logits l1, l2, l3;
        set_rings("native");
        bool ok = restored(l1) && restored(l2);
        ok = live_prefix().ok() && ok;
        ok = teacher_force(backend, forced, l3,
                           [&] { return backend.generate_in_place(request(), io, s); }) && ok;
        const double rep = ok ? max_abs_diff(l1, l2) : INFINITY;
        const double inp = ok ? max_abs_diff(l1, l3) : INFINITY;
        char buf[128];
        std::snprintf(buf, sizeof buf, "max|dlogit| restore-vs-restore=%.6g restore-vs-inplace=%.6g",
                      rep, inp);
        report("T0 determinism (floor off)", ok && rep == 0.0, buf);
        report("T0b in-place == restore (floor off)", ok && inp == 0.0, buf);
    }

    // ── T1 ───────────────────────────────────────────────────────────────
    {
        luce_test::ScopedEnvVar floor_on(kFloorEnv, "1");
        Logits l_native, l_zero, l_random, l_native2;
        set_rings("native");
        const bool a = restored(l_native);
        set_rings("zero");
        const bool b = restored(l_zero);
        set_rings("random");
        const bool c = restored(l_random);
        set_rings("native");
        const bool d = restored(l_native2);
        const double dn = a && d ? max_abs_diff(l_native, l_native2) : INFINITY;
        const double dz = a && b ? max_abs_diff(l_native, l_zero) : INFINITY;
        const double dr = a && c ? max_abs_diff(l_native, l_random) : INFINITY;
        char buf[200];
        std::snprintf(buf, sizeof buf,
                      "max|dlogit| native-vs-zero=%.6g native-vs-random=%.6g (native-vs-native again=%.6g)",
                      dz, dr, dn);
        report("T1 ring independence (floor on)", a && b && c && d && dz == 0.0 && dr == 0.0, buf);
    }

    // ── T2 ───────────────────────────────────────────────────────────────
    {
        Logits l_adopt, l_inplace, l_nofloor;
        bool ok = true;
        {
            luce_test::ScopedEnvVar floor_on(kFloorEnv, "1");
            ok = restored(l_adopt) && ok;
            // Live cache at s again (the restore + decode above moved it), then
            // continue in place with the same floor.
            ok = live_prefix().ok() && ok;
            ok = teacher_force(backend, forced, l_inplace,
                               [&] { return backend.generate_in_place(request(), io, s); }) && ok;
        }
        {
            luce_test::ScopedEnvVar floor_off(kFloorEnv, "0");
            ok = restored(l_nofloor) && ok;
        }
        const double parity = ok ? max_abs_diff(l_adopt, l_inplace) : INFINITY;
        const double floor_effect = ok ? max_abs_diff(l_adopt, l_nofloor) : 0.0;
        char buf[160];
        std::snprintf(buf, sizeof buf, "max|dlogit| adopt-vs-inplace=%.6g floor-vs-nofloor=%.6g",
                      parity, floor_effect);
        report("T2 parity: adopt+floor == in-place+floor", ok && parity == 0.0, buf);
        report("T2b the floor changes the logits", ok && floor_effect > 0.0, buf);
    }

    // ── T3 ───────────────────────────────────────────────────────────────
    {
        luce_test::ScopedEnvVar floor_off(kFloorEnv, "0");
        Logits l_cold, l_restore;
        set_rings("native");
        bool ok = restored(l_restore);
        ok = teacher_force(backend, forced, l_cold, [&] {
                 GenerateRequest r = request();
                 r.restore_points = {s};
                 return backend.generate(r, io);
             }) && ok;
        const double d = ok ? max_abs_diff(l_cold, l_restore) : INFINITY;
        char buf[96];
        std::snprintf(buf, sizeof buf, "max|dlogit| cold-vs-restore=%.6g", d);
        report("T3 floor off: restore == cold prefill", ok && d == 0.0, buf);
    }

    // ── T4 ───────────────────────────────────────────────────────────────
    {
        luce_test::ScopedEnvVar floor_on(kFloorEnv, "1");
        const auto split_request = [&] {
            GenerateRequest r = request();
            r.restore_points = {s + 40, s + 80};
            return r;
        };
        const auto restored_split = [&](Logits & out) {
            return teacher_force(backend, forced, out,
                                 [&] { return backend.restore_and_generate(0, split_request(), io); });
        };
        Logits l_native, l_zero;
        set_rings("native");
        const bool a = restored_split(l_native);
        set_rings("zero");
        const bool b = restored_split(l_zero);
        set_rings("native");
        const double dz = a && b ? max_abs_diff(l_native, l_zero) : INFINITY;
        char buf[128];
        std::snprintf(buf, sizeof buf, "max|dlogit| native-vs-zero=%.6g (restore points at s+40, s+80)", dz);
        report("T4 ring independence across several chunks", a && b && dz == 0.0, buf);
    }

    std::fprintf(stderr, "%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASS", g_failures,
                 g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
