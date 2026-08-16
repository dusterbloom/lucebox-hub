#pragma once

#include <cstddef>

// Copies one compact, natural-indexed set of Kimi expert slabs into an
// already-allocated full-width device expert. The launcher zeros every full
// destination first and completes before returning. `compact_host` contains a
// fixed metadata prefix of uint16_t natural slab indices followed by fixed-size
// gate/up/down records.
bool kimi_k3_sparse_scatter_upload(
    void * gate_device, size_t gate_full_bytes,
    void * up_device, size_t up_full_bytes,
    void * down_device, size_t down_full_bytes,
    void * compact_device, size_t compact_capacity,
    const void * compact_host, size_t compact_bytes,
    int slab_count, size_t metadata_bytes,
    size_t gate_slab_bytes, size_t up_slab_bytes,
    size_t down_slab_bytes, size_t down_slab_row_bytes,
    size_t down_full_row_bytes, int output_dim,
    const char ** failure_reason);
