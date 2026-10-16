import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kvlib import Server, chat, ROOT


def used():
    """GPU memory.used in MiB (the whole device: run under the 3090 lock with nothing else on it)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


base = used()
with Server(ROOT + "/server.log") as s:
    a = used()
    chat([{"role": "user", "content": "Say hello."}], max_tokens=64)
    b = used()
    print(f"{sys.argv[1]}: idle baseline {base} MiB; server adds {a - base} MiB after load, {b - base} MiB after one request")
