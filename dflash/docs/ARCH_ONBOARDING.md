# Architecture Onboarding — dflash Per-Arch Contract

## Purpose

This document defines the required file layout and capability declaration pattern for adding a new model architecture to dflash. When a new architecture (qwen36, future-arch-X, etc.) lands, it must follow this contract exactly. Files outside this layout in `dflash/src/<arch>/` are drift and will be flagged in review. Reviewers will reject PRs that deviate without justification. See PR #197 ("dflash: uplift DFlash runtime from qwen35/ to common/") as the live reference for the per-arch adapter pattern.

## Required Files for a New Architecture `<arch>`

Every architecture must provide this core file set in `dflash/src/<arch>/`:

| File | Purpose | Notes |
|------|---------|-------|
| `<arch>_backend.{h,cpp}` | Subclass of `ModelBackend` (defined in `common/model_backend.h`) | Entry point for architecture-specific inference. Must implement `load(gguf_path)` and declare capability virtuals. |
| `<arch>_daemon.{h,cpp}` | Daemon entry point that wires the backend into `run_daemon` from `common/daemon_loop.h` | Implements the event loop, GGUF loading, and model lifecycle. |
| `<arch>_loader.cpp` | GGUF loader responsible for deserializing model weights | Must use `common/gguf_mmap.h` and `common/gguf_metadata.h` when present; see drift-cleanup notes below. |
| `<arch>_target_graph.cpp` | Forward graph builder; constructs compute graph and tensor operations | Defines the inference pipeline shape. |

## Optional Adapters (When Architecture Supports a Capability)

Provide these only if your architecture implements the corresponding capability:

| File | Capability | Purpose | Reference |
|------|-----------|---------|-----------|
| `<arch>_dflash_target.{h,cpp}` | DFlash speculative decoding | DFlashTarget impl; mirrors qwen35's adapter | See PR #197 |
| `<arch>_mtp.{h,cpp}` | Multi-token prediction | IMtpModule impl. Two patterns: `IExternalDrafterMtp` for chain-with-h_prev arches (Gemma4); `INativeMtp` for native-heads (Qwen3.6) | See qwen36/ for INativeMtp shape |
| `<arch>_layer_split.{h,cpp}` | Multi-GPU layer splitting | Distributes model layers across GPUs | Optional for single-GPU arches |

## Capability Declaration via ModelBackend Virtuals Only

Capabilities are declared exclusively through `ModelBackend` virtual methods. Never use global registries, string dispatch, or preprocessor flags. These are the contract virtuals:

```cpp
// In common/model_backend.h

// Returns true if this architecture supports DFlash spec-decode
virtual bool supports_dflash_spec_decode() const { return false; }

// Returns the DFlashTarget instance if supports_dflash_spec_decode() is true
virtual DFlashTarget* dflash_target() { return nullptr; }

// Returns true if this architecture supports multi-token prediction
virtual bool supports_mtp() const { return false; }

// Returns the IMtpModule instance if supports_mtp() is true
virtual IMtpModule* mtp() { return nullptr; }
```

Implementations override these in the subclass constructor or after initialization. Callers check `supports_*()` before calling the accessor. If a capability is unsupported, the accessor returns `nullptr` and the caller degrades gracefully.

## Include-Path Convention (Locked)

Single include root. Every internal include is relative to that root, never via `../` and never absolute.

**Correct includes:**
```cpp
#include "common/dflash_target.h"
#include "qwen35/qwen35_dflash_target.h"
#include "common/model_backend.h"
```

**Wrong includes:**
```cpp
#include "../common/dflash_target.h"                          // relative path, forbidden
#include "/home/.../dflash_target.h"                          // absolute path, forbidden
#include "dflash/src/common/dflash_target.h"                  // redundant prefix, forbidden
```

The include root is `dflash/src/` today, set by CMake `target_include_directories`. When the upcoming `server/` + `optimization/` reorganization lands, only the CMake line will move — source files do not change. Any new architecture must respect this convention to remain forward-compatible with the reorg.

## Forbidden Patterns

Do not:

- Place architecture-specific code at top level `dflash/src/*.cpp`. Code must live in `dflash/src/<arch>/`.
- Include `internal.h` from new code (god-header, scheduled for split during the server/optimization reorg).
- Create capability-arm peer subdirectories (no `dflash/src/mtp/`, no `dflash/src/flashprefill/`, no `dflash/src/dflash/`). Capability arms live as interfaces in `common/` and adapters in `<arch>/`. Promoting them to peer subdirs is deferred to the upcoming reorg.
- Call `gguf_init_from_file` directly or manually manage mmap if `common/gguf_mmap.h` exists. Reuse the common abstraction; do not reinvent.

## Reference Implementations

- **qwen35/** — Most complete adapter set. Implements DFlash target, layer-split, and IPC drafter. Study this for patterns.
- **qwen3/** — Cleanest minimal architecture loader (254 LoC). Good template for a new arch that needs only core inference.
- **qwen36/** — Newest MTP-native architecture. Showcases the `INativeMtp` adapter pattern and how native heads declare the capability.

## When to Update This Document

This contract is live and versioned. Update it when:

1. PR #197 merges — update the required files table to reflect any new layout changes.
2. The `server/` + `optimization/` reorg lands — update the include-path root description.
3. A new capability is added to `ModelBackend` — add a corresponding optional adapter row.

All changes require a commit message that cites the reason (e.g., "docs: update ARCH_ONBOARDING for PR #197 merge").

---

Last reviewed: 2026-05-15 — keep current with PR #197 status.
