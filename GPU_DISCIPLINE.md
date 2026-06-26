## GPU Discipline

ONE session owns the GPU at a time.  No `dflash_server`, llama.cpp process, or
any other CUDA workload may launch without `scripts/gpu_lease.sh acquire`
succeeding first.

### Protocol

```
# Before launching any GPU process:
scripts/gpu_lease.sh acquire "$CLAUDE_CODE_SESSION_ID" "dflash_server" 18099

# While running (every ~30 s in a side loop or via watch):
scripts/gpu_lease.sh heartbeat "$CLAUDE_CODE_SESSION_ID"

# After killing the server:
scripts/gpu_lease.sh release "$CLAUDE_CODE_SESSION_ID"

# Check current state before doing anything:
scripts/gpu_lease.sh status
```

### Rules

1. **Check before launch** — run `status` first; if HELD, stop and coordinate
   with the other session.
2. **Acquire is atomic** — uses `flock` on `/tmp/lucebox_gpu.lease.lock`; two
   simultaneous `acquire` calls never both succeed.
3. **Heartbeat or lose it** — a lease not heartbeated within 120 s is
   auto-reclaimable.  If your server is running, ping every ~30 s.
4. **Release on done** — kill your server first, then release the lease.
   No orphan processes left behind.
5. **Never bypass** — no launching via tmux, direct shell, or background job
   without the lease.  Not even "quickly".
6. **Stale reclaim is automatic** — if the prior owner's pid is dead or
   heartbeat is >120 s old, `acquire` reclaims the lease and logs `RECLAIMED`.

### Lease file

`/tmp/lucebox_gpu.lease` — key=value, flock-protected writes.

### Selftest

```
scripts/gpu_lease.sh selftest
```

All scenarios must PASS before trusting the protocol after system changes.
