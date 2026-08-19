bool p44_compact_cache_integration_sentinel_impl(
        ggml_backend_t backend, std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
    (void) backend;
    if (err) *err = "P44 integration sentinel requires CUDA or HIP";
    return false;
#else
    constexpr int input_dim = 256;
    constexpr int output_dim = 64;
    constexpr ggml_type type = GGML_TYPE_IQ1_S;
    const std::array<uint16_t, kSlabCount> order = {
        5, 1, 9, 0, 7, 3, 11, 2, 8, 4, 10, 6};
    MoeStreamExpertSpec spec;
    spec.input_dim = input_dim;
    spec.intermediate_dim = kSlabCount * kSlabSize;
    spec.output_dim = output_dim;
    spec.gate_type = type;
    spec.up_type = type;
    spec.down_type = type;
    spec.gated_activation = MoeGatedActivation::Situ;
    ggml_quantize_init(type);
    const size_t input_row_bytes = ggml_row_size(type, input_dim);
    const size_t down_row_bytes = ggml_row_size(type, kSlabSize);
    const size_t gate_slab_bytes = input_row_bytes * kSlabSize;
    const size_t up_slab_bytes = gate_slab_bytes;
    const size_t down_slab_bytes = down_row_bytes * output_dim;
    std::vector<float> importance(input_dim, 1.0f);
    const auto quantize = [&](std::vector<uint8_t> & destination, int rows,
                              int columns, int salt) {
        const size_t row_bytes = ggml_row_size(type, columns);
        destination.resize(row_bytes * static_cast<size_t>(rows));
        std::vector<float> row(static_cast<size_t>(columns));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                row[static_cast<size_t>(c)] =
                    static_cast<float>(((r * 13 + c * 7 + salt) % 29) - 14) /
                    16.0f;
            }
            if (ggml_quantize_chunk(
                    type, row.data(), destination.data() +
                        static_cast<size_t>(r) * row_bytes,
                    0, 1, columns, importance.data()) != row_bytes) {
                return false;
            }
        }
        return true;
    };
    std::vector<SparseSlabPayload> all;
    std::vector<SparseSlabPayload> alternate;
    all.reserve(kSlabCount);
    alternate.reserve(kSlabCount);
    for (int rank = 0; rank < kSlabCount; ++rank) {
        SparseSlabPayload slab;
        slab.natural = order[static_cast<size_t>(rank)];
        if (!quantize(slab.gate, kSlabSize, input_dim, rank * 37 + 1) ||
            !quantize(slab.up, kSlabSize, input_dim, rank * 37 + 2) ||
            !quantize(slab.down, output_dim, kSlabSize, rank * 37 + 3)) {
            if (err) *err = "P44 sentinel quantization failed";
            return false;
        }
        all.push_back(std::move(slab));
        SparseSlabPayload alternate_slab;
        alternate_slab.natural = order[static_cast<size_t>(rank)];
        if (!quantize(alternate_slab.gate, kSlabSize, input_dim,
                      rank * 41 + 101) ||
            !quantize(alternate_slab.up, kSlabSize, input_dim,
                      rank * 41 + 102) ||
            !quantize(alternate_slab.down, output_dim, kSlabSize,
                      rank * 41 + 103)) {
            if (err) *err = "P44 sentinel alternate quantization failed";
            return false;
        }
        alternate.push_back(std::move(alternate_slab));
    }
    const size_t payload_bytes = gate_slab_bytes + up_slab_bytes +
        down_slab_bytes;
    if (payload_bytes > (std::numeric_limits<size_t>::max() - 4096) /
            kSlabCount) {
        if (err) *err = "P44 sentinel cache geometry overflow";
        return false;
    }
    MoeStreamExpertSpec larger_spec = spec;
    larger_spec.output_dim = kDimension;
    size_t small_slot_bytes = 0;
    size_t large_slot_bytes = 0;
    if (!kimi_k3_compact_cache_slot_bytes(
            backend, spec, small_slot_bytes, err) ||
        !kimi_k3_compact_cache_slot_bytes(
            backend, larger_spec, large_slot_bytes, err) ||
        large_slot_bytes <= small_slot_bytes ||
        large_slot_bytes > std::numeric_limits<size_t>::max() / 4) {
        if (err && err->empty()) *err = "P44 mixed-spec layout preflight failed";
        return false;
    }
    MoeStreamConfig config;
    config.device_slots = 2;
    config.device_cache_bytes = large_slot_bytes * 4;
    config.graph_cache_entries = 0;
    MoeHybridStreamEngine engine;
    if (!engine.init(backend, small_slot_bytes, config, err)) return false;
    const int small_slot_count = engine.device_slot_count();
    if (!engine.reserve_device_slot_capacity(large_slot_bytes, err) ||
        engine.device_slot_capacity() < large_slot_bytes ||
        engine.device_slot_count() >= small_slot_count ||
        engine.device_cache_bytes() > config.device_cache_bytes ||
        !engine.reserve_device_slot_capacity(large_slot_bytes, err)) {
        if (err && err->empty()) *err = "P44 mixed-spec cache reserve failed";
        return false;
    }
    SparseDeviceExpertEvaluator cached;
    SparseDeviceExpertEvaluator teacher;
    MoeStreamExternalKey key;
    key.source_domain = kCompactPrefixCacheDomain;
    key.source_generation = 0x5044343453454e54ULL;
    key.layer = 17;
    key.expert = 23;
    key.spec = spec;
    std::vector<float> input(input_dim);
    std::vector<float> reuse_input(input_dim);
    for (int i = 0; i < input_dim; ++i) {
        input[static_cast<size_t>(i)] =
            static_cast<float>((i % 17) - 8) / 9.0f;
        reuse_input[static_cast<size_t>(i)] = -input[static_cast<size_t>(i)];
    }
    struct Staging {
        ~Staging() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (context) ggml_free(context);
        }
        ggml_context * context = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor * tensor = nullptr;
    } staging;
    ggml_init_params params{};
    params.mem_size = 4 * ggml_tensor_overhead() + 1024;
    params.no_alloc = true;
    staging.context = ggml_init(params);
    if (staging.context) {
        staging.tensor = ggml_new_tensor_1d(
            staging.context, GGML_TYPE_F32, output_dim);
        staging.buffer = ggml_backend_alloc_ctx_tensors(
            staging.context, backend);
    }
    if (!staging.context || !staging.tensor || !staging.buffer) {
        if (err) *err = "P44 sentinel output staging allocation failed";
        return false;
    }
    const auto mask_for_depth = [&](int depth) {
        uint16_t mask = 0;
        for (int rank = 0; rank < depth; ++rank) {
            mask = static_cast<uint16_t>(
                mask | (1u << order[static_cast<size_t>(rank)]));
        }
        return mask;
    };
    const auto payload_range = [&](int begin, int end) {
        return std::vector<SparseSlabPayload>(
            all.begin() + begin, all.begin() + end);
    };
    const auto evaluate_owned = [&] (
            SparseDeviceExpertEvaluator & evaluator,
            const std::vector<SparseSlabPayload> & source,
            const std::vector<float> & values, int depth,
            ggml_tensor *& output) {
        std::vector<SparseSlabPayload> payload(
            source.begin(), source.begin() + depth);
        uint64_t weight_h2d = 0;
        uint64_t metadata_h2d = 0;
        bool invalid = false;
        return evaluator.evaluate_compact_device(
            backend, spec, values.data(), payload, nullptr,
            mask_for_depth(depth), down_row_bytes, output,
            weight_h2d, metadata_h2d, invalid, err) && !invalid;
    };
    const auto snapshot_slot = [&] (
            const MoeStreamExternalLease & lease, size_t bytes,
            std::vector<uint8_t> & snapshot) {
        ggml_init_params probe_params{};
        probe_params.mem_size = 2 * ggml_tensor_overhead() + 512;
        probe_params.no_alloc = true;
        ggml_context * context = ggml_init(probe_params);
        ggml_tensor * probe = context
            ? ggml_new_tensor_1d(context, GGML_TYPE_I8, bytes) : nullptr;
        const bool bound = probe && lease.bind_tensor(probe, 0, err);
        if (bound) {
            snapshot.resize(bytes);
            ggml_backend_tensor_get(probe, snapshot.data(), 0, bytes);
        }
        if (context) ggml_free(context);
        return bound;
    };
    int resident_depth = 0;
    for (const int requested_depth : {3, 5, 3, 12}) {
        ggml_tensor * reference_tensor = nullptr;
        if (!evaluate_owned(
                teacher, all, input, requested_depth, reference_tensor)) {
            return false;
        }
        std::vector<float> reference(output_dim);
        ggml_backend_tensor_get(
            reference_tensor, reference.data(), 0,
            reference.size() * sizeof(float));
        MoeStreamExternalLease lease;
        size_t required_bytes = 0;
        const uint16_t requested_mask = mask_for_depth(requested_depth);
        if (!cached.acquire_compact_cache_lease(
                backend, spec, engine, key, requested_mask, lease,
                required_bytes, err) || !lease || required_bytes == 0 ||
            lease.resident_mask() != mask_for_depth(resident_depth) ||
            lease.cache_hit() != (requested_depth <= resident_depth) ||
            lease.clear_required() != (resident_depth == 0)) {
            if (err && err->empty()) *err = "P44 sentinel lease state mismatch";
            return false;
        }
        std::vector<SparseSlabPayload> suffix = payload_range(
            std::min(resident_depth, requested_depth), requested_depth);
        ggml_tensor * cached_output = nullptr;
        uint64_t weight_h2d = 0;
        uint64_t metadata_h2d = 0;
        bool invalid = false;
        SparseDeviceExpertEvaluator::CompactCacheBinding binding{
            order.data(), &lease};
        if (!cached.evaluate_compact_device(
                backend, spec, input.data(), suffix, nullptr,
                requested_mask, down_row_bytes, cached_output,
                weight_h2d, metadata_h2d, invalid, err, &binding) ||
            invalid || !cached_output ||
            !cached.sentinel_compact_entry_restored(
                backend, spec, std::max(resident_depth, requested_depth))) {
            return false;
        }
        const uint64_t expected_h2d = static_cast<uint64_t>(
            std::max(0, requested_depth - resident_depth)) * payload_bytes;
        if (weight_h2d != expected_h2d) {
            if (err) *err = "P44 sentinel uploaded more than the suffix";
            return false;
        }
        resident_depth = std::max(resident_depth, requested_depth);
        if (lease.resident_mask() != mask_for_depth(resident_depth)) {
            if (err) *err = "P44 sentinel commit did not publish the prefix";
            return false;
        }
        std::vector<float> direct_cached(output_dim);
        ggml_backend_tensor_get(
            cached_output, direct_cached.data(), 0,
            direct_cached.size() * sizeof(float));
        if (std::memcmp(direct_cached.data(), reference.data(),
                        reference.size() * sizeof(float)) != 0) {
            if (err) *err = "P44 sentinel cached output mismatch";
            return false;
        }
        std::vector<uint8_t> slot_before_owned;
        if (requested_depth == 5 && !snapshot_slot(
                lease, required_bytes, slot_before_owned)) {
            return false;
        }
        ggml_backend_tensor_copy_async(
            backend, backend, cached_output, staging.tensor);
        lease.reset();
        ggml_tensor * reused_output = nullptr;
        if (!evaluate_owned(
                cached, alternate, reuse_input, requested_depth,
                reused_output) ||
            !cached.sentinel_compact_entry_restored(
                backend, spec, requested_depth)) {
            return false;
        }
        if (requested_depth == 5) {
            MoeStreamExternalLease verify_lease;
            size_t verify_bytes = 0;
            std::vector<uint8_t> slot_after_owned;
            if (!cached.acquire_compact_cache_lease(
                    backend, spec, engine, key, requested_mask, verify_lease,
                    verify_bytes, err) || !verify_lease ||
                !verify_lease.cache_hit() || verify_bytes != required_bytes ||
                !snapshot_slot(verify_lease, verify_bytes, slot_after_owned) ||
                slot_before_owned != slot_after_owned) {
                if (err && err->empty()) {
                    *err = "P44 sentinel owned reuse changed the cache slot";
                }
                return false;
            }
        }
        ggml_tensor * reuse_reference_tensor = nullptr;
        if (!evaluate_owned(
                teacher, alternate, reuse_input, requested_depth,
                reuse_reference_tensor)) {
            return false;
        }
        std::vector<float> staged(output_dim);
        std::vector<float> reused(output_dim);
        std::vector<float> reuse_reference(output_dim);
        ggml_backend_tensor_get(
            staging.tensor, staged.data(), 0, staged.size() * sizeof(float));
        ggml_backend_tensor_get(
            reused_output, reused.data(), 0, reused.size() * sizeof(float));
        ggml_backend_tensor_get(
            reuse_reference_tensor, reuse_reference.data(), 0,
            reuse_reference.size() * sizeof(float));
        if (std::memcmp(staged.data(), reference.data(),
                        staged.size() * sizeof(float)) != 0 ||
            std::memcmp(reused.data(), reuse_reference.data(),
                        reused.size() * sizeof(float)) != 0) {
            if (err) *err = "P44 sentinel output reuse mismatch";
            return false;
        }
    }
    const auto evaluate_cached_full = [&] (
            const MoeStreamExternalKey & cache_key,
            const std::vector<SparseSlabPayload> & source,
            const std::vector<float> & values, bool expect_hit,
            std::vector<float> & result) {
        MoeStreamExternalLease lease;
        size_t required_bytes = 0;
        if (!cached.acquire_compact_cache_lease(
                backend, spec, engine, cache_key, 0x0fff, lease,
                required_bytes, err) || !lease || required_bytes == 0 ||
            lease.cache_hit() != expect_hit) {
            return false;
        }
        std::vector<SparseSlabPayload> payload;
        if (!expect_hit) payload.assign(source.begin(), source.end());
        ggml_tensor * output = nullptr;
        uint64_t weight_h2d = 0;
        uint64_t metadata_h2d = 0;
        bool invalid = false;
        SparseDeviceExpertEvaluator::CompactCacheBinding binding{
            order.data(), &lease};
        if (!cached.evaluate_compact_device(
                backend, spec, values.data(), payload, nullptr, 0x0fff,
                down_row_bytes, output, weight_h2d, metadata_h2d, invalid,
                err, &binding) || invalid || !output ||
            weight_h2d != (expect_hit ? 0 : kSlabCount * payload_bytes) ||
            !cached.sentinel_compact_entry_restored(
                backend, spec, kSlabCount)) {
            return false;
        }
        result.resize(output_dim);
        ggml_backend_tensor_get(
            output, result.data(), 0, result.size() * sizeof(float));
        return true;
    };
    MoeStreamExternalKey alternate_key = key;
    ++alternate_key.expert;
    ggml_tensor * alternate_reference_tensor = nullptr;
    ggml_tensor * original_reference_tensor = nullptr;
    if (!evaluate_owned(
            teacher, alternate, reuse_input, kSlabCount,
            alternate_reference_tensor)) {
        return false;
    }
    std::vector<float> alternate_reference(output_dim);
    ggml_backend_tensor_get(
        alternate_reference_tensor, alternate_reference.data(), 0,
        alternate_reference.size() * sizeof(float));
    if (!evaluate_owned(
            teacher, all, input, kSlabCount, original_reference_tensor)) {
        return false;
    }
    std::vector<float> original_reference(output_dim);
    ggml_backend_tensor_get(
        original_reference_tensor, original_reference.data(), 0,
        original_reference.size() * sizeof(float));
    std::vector<float> alternate_result;
    std::vector<float> original_result;
    if (!evaluate_cached_full(
            alternate_key, alternate, reuse_input, false, alternate_result) ||
        !evaluate_cached_full(key, all, input, true, original_result) ||
        alternate_result != alternate_reference ||
        original_result != original_reference) {
        if (err && err->empty()) {
            *err = "P44 sentinel A-to-B cache rebind mismatch";
        }
        return false;
    }
    MoeStreamExternalKey larger_key = key;
    larger_key.expert += 2;
    larger_key.spec = larger_spec;
    MoeStreamExternalLease larger_lease;
    size_t larger_required_bytes = 0;
    if (!cached.acquire_compact_cache_lease(
            backend, larger_spec, engine, larger_key, mask_for_depth(3),
            larger_lease, larger_required_bytes, err) || !larger_lease ||
        larger_required_bytes != large_slot_bytes ||
        larger_lease.cache_hit() || !larger_lease.clear_required()) {
        if (err && err->empty()) *err = "P44 mixed-spec cache admission failed";
        return false;
    }
    ggml_init_params large_params{};
    large_params.mem_size = 4 * ggml_tensor_overhead() + 1024;
    large_params.no_alloc = true;
    ggml_context * large_ctx = ggml_init(large_params);
    ggml_tensor * large_gate = large_ctx ? ggml_new_tensor_2d(
        large_ctx, larger_spec.gate_type, larger_spec.input_dim,
        larger_spec.intermediate_dim) : nullptr;
    ggml_tensor * large_up = large_ctx ? ggml_new_tensor_2d(
        large_ctx, larger_spec.up_type, larger_spec.input_dim,
        larger_spec.intermediate_dim) : nullptr;
    ggml_tensor * large_down = large_ctx ? ggml_new_tensor_4d(
        large_ctx, larger_spec.down_type, kSlabSize,
        larger_spec.output_dim, kSlabCount, 1) : nullptr;
    const ggml_backend_buffer_type_t large_buft =
        ggml_backend_get_default_buffer_type(backend);
    const size_t large_alignment = large_buft
        ? std::max<size_t>(256, ggml_backend_buft_get_alignment(large_buft)) : 0;
    size_t large_cursor = 0;
    const auto bind_large = [&](ggml_tensor * tensor) {
        if (!tensor || large_alignment == 0 || large_cursor >
                std::numeric_limits<size_t>::max() - (large_alignment - 1)) {
            return false;
        }
        const size_t offset = ((large_cursor + large_alignment - 1) /
            large_alignment) * large_alignment;
        const size_t bytes = ggml_backend_buft_get_alloc_size(
            large_buft, tensor);
        if (bytes == 0 || bytes >
                std::numeric_limits<size_t>::max() - offset ||
            !larger_lease.bind_tensor(tensor, offset, err)) return false;
        large_cursor = offset + bytes;
        return true;
    };
    const bool large_bound = bind_large(large_gate) && bind_large(large_up) &&
        bind_large(large_down) && large_cursor <=
            std::numeric_limits<size_t>::max() - (large_alignment - 1) &&
        ((large_cursor + large_alignment - 1) / large_alignment) *
            large_alignment == large_slot_bytes &&
        larger_lease.clear_prefix(large_slot_bytes, err) &&
        larger_lease.commit(mask_for_depth(3), err);
    larger_lease.reset();
    if (large_ctx) ggml_free(large_ctx);
    if (!large_bound) return false;
    return true;
#endif
}
