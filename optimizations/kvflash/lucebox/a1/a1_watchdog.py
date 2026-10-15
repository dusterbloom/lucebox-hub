"""Memory watchdog (PREREG section 3): `free -m` every 10 s into <out>/mem.log; SIGTERM the run when used >= 120 GiB or swap > 1 GiB.
  python3 a1_watchdog.py <pid> <out_dir>"""
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a1_core as core  # noqa: E402


def main(pid, out):
    with open(os.path.join(out, "mem.log"), "a", buffering=1) as f:
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                return 0
            mem = core.parse_free(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
            why = core.watchdog_reason(mem)
            f.write(f"{time.strftime('%F %T')} used {mem['used']} avail {mem['available']} swap {mem['swap_used']}"
                    f"{' KILL ' + why if why else ''}\n")
            if why:
                os.kill(pid, signal.SIGTERM)
                return 1
            time.sleep(10)


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), sys.argv[2]))
