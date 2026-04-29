"""Phase A: single-point prefix cache.

Auto-detects the system-prompt boundary in token id streams via Qwen chat
template markers, hashes prefixes, and maintains an LRU map of hash → daemon
slot id. Daemon owns slot buffers; Python is the index.

Usage:
    bus = DaemonStdoutBus(daemon_proc.stdout)
    bus.start(loop)

    pc = PrefixCache(
        daemon_stdin=daemon_proc.stdin,
        await_reply=bus.await_reply,
        daemon_lock=lock,
        tokenizer=tokenizer,
        cap=4,
    )
    await pc.startup_sync()  # free orphaned slots from a previous daemon run

    # Per request (caller holds daemon_lock):
    hit = pc.lookup(prompt_ids, kv_k_type, fa_window)   # (slot_id, prefix_len) or None
    if hit:
        slot, prefix_len = hit
        # send "RESTORE <slot> <prompt_bin> <n_gen>" instead of bare line
        ...
    else:
        # send bare "<prompt_bin> <n_gen>"
        ...
        # after daemon finishes, snapshot for future cache hits:
        await pc.maybe_snapshot(prompt_ids, kv_k_type, fa_window)
"""
import asyncio
import hashlib
import struct
from collections import OrderedDict


# ---------------------------------------------------------------------------
# DaemonStdoutBus
# ---------------------------------------------------------------------------

class DaemonStdoutBus:
    """Owns the read loop on daemon stdout.

    Lines that start with a registered prefix are routed to the waiting
    coroutine; everything else is printed as a log (with noise filtering).
    """

    # Prefixes that are too spammy to print in normal operation.
    _SUPPRESS_PREFIXES = (
        "[step ", "[timing]", "[dflash]", "[prompt]",
        "[prefill]", "[migrate]", "[dbg ", "  ",
    )

    def __init__(self, stdout):
        self.stdout = stdout
        self._waiters: list[tuple[str, asyncio.Future]] = []
        self._task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._task = loop.create_task(self._run())

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, self.stdout.readline)
            if not line:
                # Daemon exited — wake all waiters with an error.
                for _, fut in self._waiters:
                    if not fut.done():
                        fut.set_exception(EOFError("daemon stdout closed"))
                self._waiters.clear()
                return
            decoded = line.decode("utf-8", errors="replace").rstrip()

            # Try to satisfy a waiter first.
            matched = False
            for i, (prefix, fut) in enumerate(self._waiters):
                if decoded.startswith(prefix) and not fut.done():
                    fut.set_result(decoded)
                    self._waiters.pop(i)
                    matched = True
                    break

            if not matched:
                # Log line — suppress very noisy prefixes.
                if decoded and not any(decoded.startswith(p) for p in self._SUPPRESS_PREFIXES):
                    print(f"  [daemon] {decoded}", flush=True)

    async def await_reply(self, prefix: str, timeout: float = 10.0) -> str:
        """Block until daemon emits a line starting with *prefix*."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._waiters.append((prefix, fut))
        return await asyncio.wait_for(fut, timeout=timeout)


# ---------------------------------------------------------------------------
# Qwen chat template helpers
# ---------------------------------------------------------------------------

def _qwen_marker_ids(tokenizer):
    """Resolve <|im_end|>, <|im_start|>, and 'system' token ids."""
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    system_t = tokenizer.encode("system", add_special_tokens=False)
    if len(im_end) != 1 or len(im_start) != 1:
        raise ValueError(
            f"Expected single-token chat markers; got "
            f"im_end={im_end} im_start={im_start}"
        )
    return im_end[0], im_start[0], system_t[0] if len(system_t) == 1 else None


def find_prefix_boundary(ids, im_end_id, im_start_id, system_token_id):
    """Return the index AFTER the FIRST end-of-system-message marker, or -1.

    Qwen's chat template renders to:

        <|im_start|>system\\nCONTENT<|im_end|>\\n<|im_start|>user\\n...

    so a `\\n` token sits BETWEEN ``<|im_end|>`` and the next ``<|im_start|>``.
    We allow up to 2 intervening tokens (covers `\\n` and similar separators).

    The cacheable prefix is the SYSTEM message: from index 0 through and
    including the ``<|im_start|>`` that begins the next role. Subsequent turns
    sharing this system message hash to the same key.

    Returns the index right after that ``<|im_start|>``, so ``ids[:boundary]``
    is the cached state and ``ids[boundary:]`` is the per-request suffix.
    Returns -1 if there is no recognizable system message.
    """
    # Find the first <|im_start|>system sequence.
    sys_idx = -1
    for i in range(len(ids) - 1):
        if ids[i] == im_start_id:
            if system_token_id is None or ids[i + 1] == system_token_id:
                sys_idx = i
                break
    if sys_idx < 0:
        return -1

    # Find the FIRST <|im_end|> after sys_idx, then locate the next <|im_start|>
    # within a small lookahead (handles a single-token newline separator).
    for i in range(sys_idx + 1, len(ids)):
        if ids[i] == im_end_id:
            for j in range(i + 1, min(i + 3, len(ids))):
                if ids[j] == im_start_id:
                    return j + 1   # boundary is one past <|im_start|>
            return -1   # malformed — im_end without subsequent im_start
    return -1


def hash_prefix(prefix_ids, kv_k_type, fa_window):
    """Stable SHA-1 (truncated 16 B) of (token ids, kv type, fa window)."""
    h = hashlib.sha1()
    h.update(struct.pack("<I", len(prefix_ids)))
    h.update(struct.pack(f"<{len(prefix_ids)}i", *prefix_ids))
    h.update(str(kv_k_type).encode())
    h.update(b"\x00")
    h.update(struct.pack("<I", fa_window or 0))
    return h.digest()[:16]


# ---------------------------------------------------------------------------
# PrefixCache
# ---------------------------------------------------------------------------

class PrefixCache:
    """LRU prefix cache.  Daemon owns the GPU slots; Python tracks hash→slot.

    Parameters
    ----------
    daemon_stdin:
        The ``stdin`` pipe of the daemon subprocess (``subprocess.Popen.stdin``).
    await_reply:
        Async callable ``(prefix: str, timeout: float) -> str`` — provided by
        ``DaemonStdoutBus.await_reply``.
    daemon_lock:
        ``asyncio.Lock`` that serialises all stdin writes + stdout reads.
        Callers must acquire it before calling ``lookup`` and hold it through
        any subsequent ``RESTORE`` / ``SNAPSHOT`` IPC.
    tokenizer:
        HuggingFace tokenizer (used only to resolve Qwen chat marker ids).
    cap:
        Maximum number of snapshot slots.  0 disables the cache entirely.
    log_prefix:
        String prepended to cache-hit/miss log lines.
    """

    # Daemon-side hard cap (PREFIX_CACHE_SLOTS in test_dflash.cpp). Any
    # configured cap > this is silently clamped down — exceeding it would
    # cause silent SNAPSHOT failures on slots ≥ 8.
    DAEMON_MAX_SLOTS = 8

    def __init__(self, *, daemon_stdin, await_reply, daemon_lock,
                 tokenizer, kv_k_type: str, fa_window: int,
                 cap: int = 4, log_prefix: str = "[pc]"):
        self.stdin = daemon_stdin
        self._await_reply = await_reply
        self.lock = daemon_lock
        self.log_prefix = log_prefix
        # Cache key fields — fixed at daemon spawn (env vars passed through).
        # Mismatched values across turns are not possible within one server
        # process, but they're still part of the hash so a daemon restart
        # with different flags doesn't return stale state.
        self.kv_k_type = kv_k_type
        self.fa_window = fa_window

        if cap > self.DAEMON_MAX_SLOTS:
            print(f"{log_prefix} cap={cap} exceeds daemon limit "
                  f"({self.DAEMON_MAX_SLOTS}); clamping", flush=True)
            cap = self.DAEMON_MAX_SLOTS
        self.cap = cap

        if cap <= 0:
            self.disabled = True
            return
        self.disabled = False

        self.entries: OrderedDict[bytes, int] = OrderedDict()  # hash → slot_id
        self.next_slot = 0
        self.im_end, self.im_start, self.system_t = _qwen_marker_ids(tokenizer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def boundary(self, ids: list[int]) -> int:
        if self.disabled:
            return -1
        return find_prefix_boundary(ids, self.im_end, self.im_start, self.system_t)

    def lookup(self, prompt_ids: list[int]) -> tuple[int, int] | None:
        """Return ``(slot_id, prefix_len)`` on cache hit, else ``None``.

        The caller must already hold ``daemon_lock`` before inspecting the
        returned slot, since the slot id may be evicted by a concurrent
        request otherwise.
        """
        if self.disabled:
            return None
        b = self.boundary(prompt_ids)
        if b <= 0:
            return None
        key = hash_prefix(prompt_ids[:b], self.kv_k_type, self.fa_window)
        if key in self.entries:
            self.entries.move_to_end(key)   # mark fresh
            return self.entries[key], b
        return None

    async def maybe_snapshot(self, prompt_ids: list[int],
                              token_stream_consumer=None) -> None:
        """Snapshot the daemon's KV state at the cacheable prefix boundary.

        Implementation pattern: rather than try to take a snapshot at end-of-
        generation (where ``cache.cur_pos`` is well past the prefix boundary),
        we issue a SECOND prefill pass of the prefix-only token stream with
        ``n_gen=0``. This costs one extra system-prompt prefill on cold turns
        but guarantees the snapshot's ``cur_pos`` exactly matches the
        cache-key prefix length. Subsequent turns hit the cache and skip the
        whole system-prompt prefill, recovering the cost many times over.

        Caller must hold ``daemon_lock``.  ``token_stream_consumer`` is an
        async callable (or None) that drains the daemon's stream-fd token
        output for the prefill pass; pass the same drainer as the request
        handler so the ``-1`` sentinel is consumed cleanly.
        """
        if self.disabled:
            return
        b = self.boundary(prompt_ids)
        if b <= 0:
            return
        key = hash_prefix(prompt_ids[:b], self.kv_k_type, self.fa_window)
        if key in self.entries:
            return  # already cached

        # Evict LRU entry if at capacity.
        if len(self.entries) >= self.cap:
            old_key, old_slot = self.entries.popitem(last=False)
            self._send(f"FREE_SNAPSHOT {old_slot}\n")
            await self._await_reply("[snap] freed slot=")
            slot = old_slot
        else:
            slot = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.cap

        # Write the prefix-only tokens to a temp file and prefill them with
        # n_gen=0 so the daemon ends with cur_pos == prefix length.
        import os, struct, tempfile
        fd, tmp_path = tempfile.mkstemp(suffix="_prefix.bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids[:b]:
                f.write(struct.pack("<i", int(t)))
        try:
            self._send(f"{tmp_path} 0\n")
            # Drain the prefill's token stream (just the -1 sentinel since
            # n_gen=0 means no real tokens). Caller-supplied drainer.
            if token_stream_consumer is not None:
                await token_stream_consumer()

            self._send(f"SNAPSHOT {slot}\n")
            await self._await_reply("[snap] slot=")
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass

        self.entries[key] = slot
        print(f"{self.log_prefix} snapshot slot={slot} prefix_len={b}", flush=True)

    async def startup_sync(self) -> None:
        """Query the daemon for existing slots and free them all.

        Called once at server startup to ensure Python's hash table is
        consistent with the daemon's slot state (both empty after this).
        """
        if self.disabled:
            return
        async with self.lock:
            self._send("LIST_SLOTS\n")
            reply = await self._await_reply("[snap] slots=")
            slots_str = reply.split("[snap] slots=", 1)[1].strip()
            if not slots_str:
                return
            orphans = [s.strip() for s in slots_str.split(",") if s.strip()]
            for s in orphans:
                self._send(f"FREE_SNAPSHOT {s}\n")
                await self._await_reply("[snap] freed slot=")
            print(f"{self.log_prefix} freed {len(orphans)} orphaned daemon slots",
                  flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, line: str) -> None:
        self.stdin.write(line.encode("utf-8"))
        self.stdin.flush()
