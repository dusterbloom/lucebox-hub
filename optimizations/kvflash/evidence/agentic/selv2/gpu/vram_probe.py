"""Peak VRAM of the v2 server (E2E config) on one ~44K-token request; free VRAM left for a GPU scorer.
usage: python3 -I vram_probe.py <log path>   (run under the 3090 lock)"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BIN = "/mnt/kimi-k3/build/selv2-cuda/luce_server"
ARGS = [BIN, "/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf", "--host", "127.0.0.1", "--port", "18291",
        "--max-ctx", "52224", "--model-name", "luce", "--chat-template-file",
        "/mnt/kimi-k3/artifacts/kvflash-am-gpu/tmpl/qwen36_template.jinja", "--prefix-cache-max-mib", "12288",
        "--prefix-cache-slots", "4", "--kvflash", "16384"]
ENV = dict(os.environ, LUCE_KVFLASH_PREFILL_RESELECT="4", LUCE_KVFLASH_RESELECT_SCORER="exact", LUCE_KVFLASH_RECALL_BLOCKS="6",
           LUCE_KVFLASH_SCORE_POOL_RECENT="1", LUCE_KVFLASH_SELECT_QUERIES="64", LUCE_KVFLASH_SELECT_UNIT="2",
           LUCE_KVFLASH_POLICY="qk", KVFLASH_STATE_LADDER="1", KVFLASH_LIVE_RESTORE="1",
           TMPDIR="/home/peppi/.claude/jobs/697c93e7/tmp")   # as run_arm.py QK16KND
URL = "http://127.0.0.1:18291"


def smi():
    # per-process used_memory is N/A on WSL; memory.used sometimes overflows (garbage > total), report 0 then
    q = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    used, tot = (int(x) for x in subprocess.run(q, capture_output=True, text=True).stdout.split(","))
    return (used if used <= tot else 0), tot


def main(log):
    f = open(log, "w")
    p = subprocess.Popen(ARGS, stdout=f, stderr=subprocess.STDOUT, env=ENV, stdin=subprocess.DEVNULL)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(URL + "/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        print("after load: used %d of %d MiB" % smi(), flush=True)
        lines = "\n".join(f"Record {i}: item {i * 7919 % 10007} shipped to depot {i % 97} on day {i % 365}." for i in range(int(sys.argv[2])))
        body = {"model": "luce", "temperature": 0, "seed": 1, "max_tokens": 16, "messages": [
            {"role": "system", "content": "You are a careful assistant."},
            {"role": "user", "content": lines + "\n\nWhich depot got record 17? Answer with the number only."}]}
        peak, done = [0], threading.Event()

        def poll():
            while not done.is_set():
                peak[0] = max(peak[0], smi()[0])
                time.sleep(0.2)
        t = threading.Thread(target=poll, daemon=True)
        t.start()
        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        done.set()
        t.join()
        used, tot = smi()
        print("answer %r usage %s" % (r["choices"][0]["message"]["content"], r.get("usage")), flush=True)
        print("after request: used %d MiB; peak during request %d of %d MiB; free at peak %d MiB" % (used, peak[0], tot, tot - peak[0]))
    finally:
        p.terminate()
        p.wait(60)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:400])
