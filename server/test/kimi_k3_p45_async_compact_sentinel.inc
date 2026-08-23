bool p45_async_compact_sentinel_impl(
        ggml_backend_t backend, std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
    (void) backend;
    if (err) *err = "P45 async compact sentinel requires CUDA or HIP";
    return false;
#else
    constexpr int input_dim = 256;
    constexpr int output_dim = 64;
    constexpr int jobs = kNativeTopK;
    constexpr ggml_type type = GGML_TYPE_IQ1_S;
    const std::array<uint16_t, kSlabCount> order = {
        5, 1, 9, 0, 7, 3, 11, 2, 8, 4, 10, 6};
    const std::array<int, jobs> depths = {
        1, 3, 12, 5, 3, 1, 8, 12, 5, 3, 1, 8, 12, 5, 3, 1};
    MoeStreamExpertSpec spec;
    spec.input_dim = input_dim;
    spec.intermediate_dim = kSlabCount * kSlabSize;
    spec.output_dim = output_dim;
    spec.gate_type = type;
    spec.up_type = type;
    spec.down_type = type;
    spec.gated_activation = MoeGatedActivation::Situ;
    ggml_quantize_init(type);
    const size_t down_row_bytes = ggml_row_size(type, kSlabSize);
    std::vector<float> importance(input_dim, 1.0f);
    const auto quantize = [&](std::vector<uint8_t> & destination, int rows,
                              int columns, int salt) {
        const size_t row_bytes = ggml_row_size(type, columns);
        destination.resize(row_bytes * static_cast<size_t>(rows));
        std::vector<float> row(static_cast<size_t>(columns));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                row[static_cast<size_t>(c)] = static_cast<float>(
                    ((r * 13 + c * 7 + salt) % 31) - 15) / 17.0f;
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
    std::vector<float> input(static_cast<size_t>(input_dim));
    for (int column = 0; column < input_dim; ++column) {
        input[static_cast<size_t>(column)] =
            static_cast<float>(((column * 5) % 37) - 18) / 19.0f;
    }
    std::array<std::vector<SparseSlabPayload>, jobs> payloads;
    std::array<uint16_t, jobs> masks{};
    for (int job = 0; job < jobs; ++job) {
        auto & payload = payloads[static_cast<size_t>(job)];
        payload.reserve(static_cast<size_t>(depths[static_cast<size_t>(job)]));
        for (int rank = 0; rank < depths[static_cast<size_t>(job)]; ++rank) {
            SparseSlabPayload slab;
            slab.natural = order[static_cast<size_t>(rank)];
            const int salt = job * 401 + rank * 37;
            if (!quantize(slab.gate, kSlabSize, input_dim, salt + 1) ||
                !quantize(slab.up, kSlabSize, input_dim, salt + 2) ||
                !quantize(slab.down, output_dim, kSlabSize, salt + 3)) {
                if (err) *err = "P45 async sentinel quantization failed";
                return false;
            }
            masks[static_cast<size_t>(job)] = static_cast<uint16_t>(
                masks[static_cast<size_t>(job)] | (1u << slab.natural));
            payload.push_back(std::move(slab));
        }
    }

    SparseCompactPayload component_major;
    component_major.slab_count = depths[0];
    component_major.gate_slab_bytes = payloads[0][0].gate.size();
    component_major.up_slab_bytes = payloads[0][0].up.size();
    component_major.down_slab_bytes = payloads[0][0].down.size();
    component_major.component_major = true;
    KimiK3CompactWireLayout component_layout;
    uint16_t component_mask = 0;
    if (!kimi_k3_compact_wire_layout(
            component_major.slab_count, component_major.gate_slab_bytes,
            component_major.up_slab_bytes, component_major.down_slab_bytes,
            &component_layout) ||
        !component_major.ensure(component_layout.total_bytes, err)) {
        return false;
    }
    component_major.bytes = component_layout.total_bytes;
    if (!pack_sparse_component_major(
            payloads[0], nullptr, component_major.data,
            component_major.capacity, component_layout, component_mask, err) ||
        component_mask != masks[0]) {
        if (err && err->empty()) {
            *err = "P45 direct component wire construction failed";
        }
        return false;
    }
    const std::vector<uint8_t> component_snapshot(
        static_cast<const uint8_t *>(component_major.data),
        static_cast<const uint8_t *>(component_major.data) +
            component_major.bytes);

    SparseDeviceExpertEvaluator teacher;
    SparseDeviceExpertEvaluator queued;
    std::vector<float> expected(static_cast<size_t>(jobs) * output_dim);
    for (int job = 0; job < jobs; ++job) {
        ggml_tensor * output = nullptr;
        uint64_t weight_h2d = 0;
        uint64_t metadata_h2d = 0;
        bool invalid = false;
        if (!teacher.evaluate_compact_device(
                backend, spec, input.data(),
                payloads[static_cast<size_t>(job)], nullptr,
                masks[static_cast<size_t>(job)], down_row_bytes, output,
                weight_h2d, metadata_h2d, invalid, err) ||
            invalid || !output) {
            return false;
        }
        ggml_backend_tensor_get(
            output, expected.data() + static_cast<size_t>(job) * output_dim,
            0, static_cast<size_t>(output_dim) * sizeof(float));
    }

    struct OutputArena {
        ~OutputArena() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (context) ggml_free(context);
        }
        ggml_context * context = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_tensor * rows = nullptr;
        ggml_tensor * device_input = nullptr;
        std::array<ggml_tensor *, jobs> row{};
    } arena;
    ggml_init_params params{};
    params.mem_size = (jobs + 2) * ggml_tensor_overhead() + 1024;
    params.no_alloc = true;
    arena.context = ggml_init(params);
    if (arena.context) {
        arena.rows = ggml_new_tensor_2d(
            arena.context, GGML_TYPE_F32, output_dim, jobs);
        arena.device_input = ggml_new_tensor_2d(
            arena.context, GGML_TYPE_F32, input_dim, 1);
        for (int job = 0; job < jobs; ++job) {
            arena.row[static_cast<size_t>(job)] = ggml_view_1d(
                arena.context, arena.rows, output_dim,
                static_cast<size_t>(job) * output_dim * sizeof(float));
        }
        arena.buffer = ggml_backend_alloc_ctx_tensors(arena.context, backend);
    }
    if (!arena.context || !arena.rows || !arena.device_input ||
        !arena.buffer) {
        if (err) *err = "P45 async sentinel output arena allocation failed";
        return false;
    }
    ggml_backend_tensor_set(
        arena.device_input, input.data(), 0,
        input.size() * sizeof(float));
    const auto queue_job = [&](int job, ggml_tensor * destination,
                               const float * input_data) {
        ggml_tensor * output = nullptr;
        uint64_t weight_h2d = 0;
        uint64_t metadata_h2d = 0;
        bool invalid = false;
        const SparseCompactPayload * prepacked = job == 0
            ? &component_major : nullptr;
        if (!queued.evaluate_compact_device(
                backend, spec, input_data,
                payloads[static_cast<size_t>(job)], prepacked,
                masks[static_cast<size_t>(job)], down_row_bytes, output,
                weight_h2d, metadata_h2d, invalid, err) ||
            invalid || !output) {
            return false;
        }
        ggml_backend_tensor_copy_async(backend, backend, output, destination);
        return true;
    };
    if (!queued.begin_compact_async_batch(backend, jobs, err)) return false;
    for (int job = 0; job < jobs; ++job) {
        if (!queue_job(
                job, arena.row[static_cast<size_t>(job)], input.data())) {
            queued.abort_compact_async_batch();
            return false;
        }
    }
    ggml_backend_synchronize(backend);
    if (!queued.complete_compact_async_batch_after_sync(err)) return false;
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(
        arena.rows, actual.data(), 0, actual.size() * sizeof(float));
    if (std::memcmp(
            actual.data(), expected.data(), actual.size() * sizeof(float)) != 0) {
        if (err) *err = "P45 async sentinel queued output mismatch";
        return false;
    }
    if (std::memcmp(
            component_major.data, component_snapshot.data(),
            component_snapshot.size()) != 0) {
        if (err) *err = "P45 direct component wire was mutated";
        return false;
    }

    if (!queued.begin_compact_async_batch(backend, 1, err) ||
        !queue_job(0, arena.row[0], input.data())) {
        queued.abort_compact_async_batch();
        return false;
    }
    queued.abort_compact_async_batch();
    if (!queued.begin_compact_async_batch(backend, 1, err) ||
        !queue_job(0, arena.row[0], input.data())) {
        queued.abort_compact_async_batch();
        return false;
    }
    ggml_backend_synchronize(backend);
    if (!queued.complete_compact_async_batch_after_sync(err)) return false;
    std::vector<float> recovered(output_dim);
    ggml_backend_tensor_get(
        arena.row[0], recovered.data(), 0,
        recovered.size() * sizeof(float));
    if (std::memcmp(
            recovered.data(), expected.data(),
            recovered.size() * sizeof(float)) != 0) {
        if (err) *err = "P45 async sentinel recovery output mismatch";
        return false;
    }
    if (!queued.begin_compact_async_batch(
            backend, 1, err, arena.device_input) ||
        !queue_job(0, arena.row[0], nullptr)) {
        queued.abort_compact_async_batch();
        return false;
    }
    ggml_backend_synchronize(backend);
    if (!queued.complete_compact_async_batch_after_sync(err)) return false;
    ggml_backend_tensor_get(
        arena.row[0], recovered.data(), 0,
        recovered.size() * sizeof(float));
    if (std::memcmp(
            recovered.data(), expected.data(),
            recovered.size() * sizeof(float)) != 0) {
        if (err) *err = "P45 device-input output mismatch";
        return false;
    }
    const SparseDeviceExpertEvaluator::CompactAsyncStats & stats =
        queued.compact_async_stats();
    if (stats.begins != 4 || stats.jobs != jobs + 3 ||
        stats.h2d_calls != 4 * (jobs + 3) + 3 ||
        stats.input_d2d_copies != 1 ||
        stats.input_d2d_bytes != input.size() * sizeof(float) ||
        stats.graph_enqueues != jobs + 3 || stats.layer_flushes != 3 ||
        stats.abort_syncs != 1 || stats.max_inflight != jobs ||
        stats.h2d_bytes == 0 || stats.submit_ns == 0 ||
        stats.device_window_ns == 0) {
        if (err) *err = "P45 async sentinel accounting mismatch";
        return false;
    }
    return true;
#endif
}

bool compact_union_owned_buffer_sentinel_for_quant_types(
        ggml_backend_t backend, ggml_type gate_up_type,
        ggml_type down_type, std::string * err) {
#if !defined(DFLASH27B_BACKEND_CUDA) && !defined(DFLASH27B_BACKEND_HIP)
    (void) backend;
    (void) gate_up_type;
    (void) down_type;
    if (err) *err = "compact union sentinel requires CUDA or HIP";
    return false;
#else
    constexpr int input_dim = 3584;
    constexpr int output_dim = 3584;
    constexpr int max_rows = 9;
    MoeStreamExpertSpec spec;
    spec.input_dim = input_dim;
    spec.intermediate_dim = kSlabCount * kSlabSize;
    spec.output_dim = output_dim;
    spec.gate_type = gate_up_type;
    spec.up_type = gate_up_type;
    spec.down_type = down_type;
    spec.gated_activation = MoeGatedActivation::Situ;
    spec.situ_beta = 4.0f;
    spec.situ_linear_beta = 25.0f;
    ggml_quantize_init(spec.gate_type);
    ggml_quantize_init(spec.down_type);

    const auto quantize = [&](ggml_type type,
                              std::vector<uint8_t> & destination,
                              int rows, int columns, uint32_t salt) {
        const size_t row_bytes = ggml_row_size(type, columns);
        destination.resize(row_bytes * static_cast<size_t>(rows));
        std::vector<float> row(static_cast<size_t>(columns));
        std::vector<float> importance(static_cast<size_t>(columns), 1.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                uint32_t value = salt ^
                    static_cast<uint32_t>(r + 1) * 0x9e3779b9U ^
                    static_cast<uint32_t>(c + 1) * 0x85ebca6bU;
                value ^= value >> 16;
                value *= 0x7feb352dU;
                value ^= value >> 15;
                row[static_cast<size_t>(c)] =
                    0.04f * static_cast<float>(
                        static_cast<int>(value & 0xffffU) - 32768) /
                    32768.0f;
            }
            if (ggml_quantize_chunk(
                    type, row.data(),
                    destination.data() + static_cast<size_t>(r) * row_bytes,
                    0, 1, columns, importance.data()) != row_bytes) {
                return false;
            }
        }
        return true;
    };

    SparseSlabPayload slab;
    slab.natural = 5;
    if (!quantize(
            spec.gate_type, slab.gate, kSlabSize, input_dim, 0x475445U) ||
        !quantize(
            spec.up_type, slab.up, kSlabSize, input_dim, 0x555045U) ||
        !quantize(
            spec.down_type, slab.down, output_dim, kSlabSize, 0x444e45U)) {
        if (err) *err = "compact union sentinel quantization failed";
        return false;
    }
    const std::vector<SparseSlabPayload> payload = {slab};
    const size_t down_row_bytes =
        ggml_row_size(spec.down_type, kSlabSize);
    SparseCompactPayload compact;
    compact.slab_count = 1;
    compact.gate_slab_bytes = slab.gate.size();
    compact.up_slab_bytes = slab.up.size();
    compact.down_slab_bytes = slab.down.size();
    compact.component_major = true;
    KimiK3CompactWireLayout layout;
    uint16_t compact_mask = 0;
    if (!kimi_k3_compact_wire_layout(
            1, compact.gate_slab_bytes, compact.up_slab_bytes,
            compact.down_slab_bytes, &layout) ||
        !compact.ensure(layout.total_bytes, err)) {
        return false;
    }
    compact.bytes = layout.total_bytes;
    if (!pack_sparse_component_major(
            payload, nullptr, compact.data, compact.capacity, layout,
            compact_mask, err)) {
        return false;
    }

    std::vector<float> inputs(static_cast<size_t>(max_rows) * input_dim);
    for (int row = 0; row < max_rows; ++row) {
        for (int column = 0; column < input_dim; ++column) {
            inputs[static_cast<size_t>(row) * input_dim + column] =
                0.03f * std::sin(
                    0.013f * static_cast<float>(column + 1) +
                    0.17f * static_cast<float>(row + 1));
        }
    }
    std::array<uint16_t, max_rows> masks{};
    masks.fill(compact_mask);

    SparseDeviceExpertEvaluator teacher;
    std::vector<float> expected(static_cast<size_t>(max_rows) * output_dim);
    for (int row = 0; row < max_rows; ++row) {
        ggml_tensor * output = nullptr;
        uint64_t weight_h2d = 0;
        uint64_t metadata_h2d = 0;
        bool invalid = false;
        if (!teacher.evaluate_compact_device(
                backend, spec,
                inputs.data() + static_cast<size_t>(row) * input_dim,
                payload, nullptr, compact_mask, down_row_bytes, output,
                weight_h2d, metadata_h2d, invalid, err) ||
            invalid || !output) {
            return false;
        }
        ggml_backend_tensor_get(
            output,
            expected.data() + static_cast<size_t>(row) * output_dim,
            0, static_cast<size_t>(output_dim) * sizeof(float));
    }

    SparseDeviceExpertEvaluator candidate;
    if (!candidate.prepare_compact_union_outputs(
            backend, output_dim, max_rows, err)) {
        return false;
    }
    for (const int width : {1, 2, 8}) {
        const int rows = width + 1;
        const uint64_t graph_ns_before = candidate.expert_graph_ns();
        const uint64_t d2d_copies_before =
            candidate.compact_union_output_d2d_copies();
        const uint64_t d2d_bytes_before =
            candidate.compact_union_output_d2d_bytes();
        uint64_t weight_h2d = 0;
        uint64_t metadata_h2d = 0;
        uint64_t graph_ns = 0;
        bool invalid = false;
        const auto select_targets = [&](
                int base, int valid,
                std::array<ggml_tensor *, 8> & targets,
                std::string *) {
            for (int lane = 0; lane < width; ++lane) {
                targets[static_cast<size_t>(lane)] = lane < valid
                    ? candidate.compact_union_output(
                        static_cast<size_t>(base + lane))
                    : candidate.compact_union_padding_output(
                        static_cast<size_t>(lane));
            }
            return true;
        };
        if (!candidate.evaluate_compact_union_device(
                backend, spec, inputs.data(), rows, width, compact,
                masks.data(), down_row_bytes, select_targets, weight_h2d,
                metadata_h2d, graph_ns, invalid, err) || invalid) {
            return false;
        }
        if (weight_h2d != slab.gate.size() + slab.up.size() +
                slab.down.size() || metadata_h2d == 0 || graph_ns == 0 ||
            candidate.expert_graph_ns() - graph_ns_before != graph_ns ||
            candidate.compact_union_output_d2d_copies() != d2d_copies_before ||
            candidate.compact_union_output_d2d_bytes() != d2d_bytes_before) {
            if (err) *err = "compact union owned-buffer accounting mismatch";
            return false;
        }
        ggml_backend_synchronize(backend);
        std::vector<float> actual(static_cast<size_t>(rows) * output_dim);
        ggml_backend_tensor_get(
            candidate.compact_union_outputs(), actual.data(), 0,
            actual.size() * sizeof(float));
        if (std::memcmp(
                actual.data(), expected.data(),
                actual.size() * sizeof(float)) != 0) {
            if (err) *err = "compact union owned-buffer output mismatch";
            return false;
        }
    }
    return true;
#endif
}

bool compact_union_owned_buffer_sentinel_impl(
        ggml_backend_t backend, std::string * err) {
    return compact_union_owned_buffer_sentinel_for_quant_types(
               backend, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, err) &&
        compact_union_owned_buffer_sentinel_for_quant_types(
               backend, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_S, err) &&
        compact_union_owned_buffer_sentinel_for_quant_types(
               backend, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, err);
}
