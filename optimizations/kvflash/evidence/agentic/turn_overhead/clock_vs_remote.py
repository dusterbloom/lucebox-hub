"""Which local clock keeps true time? Compare local monotonic and wall elapsed against an NTP-synced remote host.
    python3 -I clock_vs_remote.py <ssh host> <seconds>"""
import subprocess
import sys
import time

host, dur = sys.argv[1], float(sys.argv[2])


def remote():
    t = time.monotonic()
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "date +%s.%N"], capture_output=True, text=True, check=True).stdout
    rtt = time.monotonic() - t
    return float(out), rtt


r0, rtt0 = remote()
m0, w0 = time.monotonic(), time.time()
time.sleep(dur)
r1, rtt1 = remote()
m1, w1 = time.monotonic(), time.time()
r, m, w = r1 - r0, m1 - m0, w1 - w0
print(f"remote {r:.3f}s  local monotonic {m:.3f}s  local wall {w:.3f}s  (ssh rtt {rtt0:.3f}, {rtt1:.3f})")
print(f"monotonic/remote {m / r:.4f}  wall/remote {w / r:.4f}")
