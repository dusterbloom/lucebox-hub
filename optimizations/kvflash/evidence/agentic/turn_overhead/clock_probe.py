"""Does the wall clock (time.time, what run_arm.py uses) jump against the monotonic clock?
    python3 -I clock_probe.py <seconds>
Prints every 0.1 s sample where the two clocks' deltas disagree by more than 50 ms, then a summary."""
import sys
import time

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
w0, m0 = time.time(), time.monotonic()
pw, pm = w0, m0
jumps = []
while time.monotonic() - m0 < dur:
    time.sleep(0.1)
    w, m = time.time(), time.monotonic()
    d = (w - pw) - (m - pm)
    if abs(d) > 0.05:
        jumps.append((round(m - m0, 2), round(d, 3)))
        print(f"t={m - m0:8.2f}s wall-minus-monotonic step {d:+.3f}s", flush=True)
    pw, pm = w, m
drift = (time.time() - w0) - (time.monotonic() - m0)
print(f"samples over {dur:.0f}s: {len(jumps)} steps over 50 ms, total drift wall-minus-monotonic {drift:+.3f}s")
