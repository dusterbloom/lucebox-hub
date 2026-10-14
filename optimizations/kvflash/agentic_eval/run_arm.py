"""Drive one engine arm over the task list. Server-side timings only (usage.timings from the response).
Usage: python3 -I run_arm.py <arm FULL|POOL8K|POOL16K|FULLB> <tasks.jsonl> <trajectories.jsonl> <out_dir> [--retained retained_probes.json]

Arms (identical GGUF, KV type default, temp 0, seed 1, --max-ctx, chat template; only the KVFlash flags differ):
  FULL    : no --kvflash, no KVFLASH_* env
  FULLB   : identical to FULL (nondeterminism / prefix-cache-order control)
  POOL8K  : --kvflash 8192  + KVFLASH_STATE_LADDER=1 KVFLASH_LIVE_RESTORE=1
  POOL16K : --kvflash 16384 + KVFLASH_STATE_LADDER=1 KVFLASH_LIVE_RESTORE=1
Probe tasks: if --retained is given, only probes whose id is listed run (the list is decided on FULL alone).
Resumable: results append to <out_dir>/<arm>.jsonl, finished ids are skipped."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert  # noqa: E402

BIN = "/mnt/kimi-k3/build/kvflash-am-cuda/luce_server"
MODEL = "/mnt/kimi-k3/home-models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_M.gguf"
MAX_CTX = 52224
PORT = 18290
URL = f"http://127.0.0.1:{PORT}"
TMPDIR = "/mnt/kimi-k3/build/tmp"
ARMS = {"FULL": None, "FULLB": None, "POOL8K": 8192, "POOL16K": 16384}
PROBE_MAX_TOKENS, DECISION_MAX_TOKENS = 64, 256
PC_MIB = 12288  # host prefix-cache budget, same on every arm
PC_SLOTS = int(os.environ.get("AE_PC_SLOTS", "4"))  # few slots: keeps the shared probe prefix from being LRU-evicted by one-shot probe leaves


def gpu_pids():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    return [x for x in r.stdout.split() if x.strip()]


def wait_idle(poll_gap=300):
    """Idle = no compute apps on 2 polls poll_gap seconds apart."""
    while True:
        if gpu_pids():
            print("GPU busy; re-poll in %ds" % poll_gap, flush=True)
            time.sleep(poll_gap)
            continue
        time.sleep(poll_gap)
        if not gpu_pids():
            return
        print("GPU became busy on 2nd poll", flush=True)


def start_server(arm, log_path):
    env = dict(os.environ)
    env["TMPDIR"] = TMPDIR
    for k in list(env):
        if k.startswith("KVFLASH_"):
            del env[k]
    args = [BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--max-ctx", str(MAX_CTX), "--model-name", "luce",
            "--chat-template-file", convert.TEMPLATE, "--prefix-cache-max-mib", str(PC_MIB), "--prefix-cache-slots", str(PC_SLOTS)]
    if ARMS[arm]:
        env.update({"KVFLASH_STATE_LADDER": "1", "KVFLASH_LIVE_RESTORE": "1"})
        args += ["--kvflash", str(ARMS[arm])]
    f = open(log_path, "a")
    f.write("### start %s %s\n" % (arm, " ".join(args)))
    f.flush()
    p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
    for _ in range(400):
        if p.poll() is not None:
            raise RuntimeError("server died at start; see " + log_path)
        try:
            urllib.request.urlopen(URL + "/v1/models", timeout=2)
            return p, f
        except Exception:
            time.sleep(2)
    raise RuntimeError("server start timeout")


def stop_server(p, f):
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(60)
        except Exception:
            p.kill()
            p.wait()
    f.close()


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def build_request(task, traj, max_tokens=None):
    msgs = list(traj["messages"][: task["cut_index"]])
    if task["kind"] == "probe":  # (primer = plain decision prompt, no question)
        msgs.append({"role": "user", "content": task["question"]})
        mt = PROBE_MAX_TOKENS
    else:
        mt = DECISION_MAX_TOKENS
    return {"model": "luce", "messages": msgs, "tools": traj["tools"], "temperature": 0, "seed": 1, "max_tokens": max_tokens or mt,
            "stream": False, "chat_template_kwargs": dict(convert.CHAT_KWARGS)}


def run_task(task, traj, max_tokens=None):
    body = build_request(task, traj, max_tokens)
    t0 = time.time()
    resp = post(body)
    wall_client = time.time() - t0
    ch = resp["choices"][0]
    m = ch["message"]
    u = resp.get("usage", {})
    return {"id": task["id"], "traj": task["traj"], "kind": task["kind"], "content": m.get("content"),
            "reasoning": (m.get("reasoning_content") or "")[:200], "tool_calls": m.get("tool_calls"),
            "finish": ch.get("finish_reason"), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "timings": u.get("timings"), "client_wall_s": round(wall_client, 3)}


def main():
    arm, tasks_p, traj_p, out_dir = sys.argv[1:5]
    retained = None
    if "--retained" in sys.argv:
        retained = set(json.load(open(sys.argv[sys.argv.index("--retained") + 1])))
    skip_wait = "--no-wait" in sys.argv
    os.makedirs(out_dir, exist_ok=True)
    trajs = {}
    for line in open(traj_p):
        r = json.loads(line)
        trajs[r["trajectory_id"]] = r
    tasks = [json.loads(l) for l in open(tasks_p)]
    tasks = [t for t in tasks if t["traj"] in trajs]
    if retained is not None:
        tasks = [t for t in tasks if t["kind"] == "decision" or t["id"] in retained]
    res_p = os.path.join(out_dir, arm + ".jsonl")
    done = set()
    if os.path.exists(res_p):
        done = {json.loads(l)["id"] for l in open(res_p)}
    todo = [t for t in tasks if t["id"] not in done]
    print(f"arm {arm}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return
    if not skip_wait:
        wait_idle()
    md5 = hashlib.md5(open(BIN, "rb").read()).hexdigest()
    json.dump({"arm": arm, "binary": BIN, "binary_md5": md5, "model": MODEL, "max_ctx": MAX_CTX, "pool": ARMS[arm],
               "started": time.strftime("%F %T")}, open(os.path.join(out_dir, arm + ".meta.json"), "w"))
    log_path = os.path.join(out_dir, arm + ".server.log")
    p, lf = start_server(arm, log_path)
    restarts = 0
    try:
        with open(res_p, "a") as out:
            last_primer = None
            for n, task in enumerate(todo):
                try:
                    key = (task["traj"], task["cut_index"])
                    if task["kind"] == "probe" and key != last_primer:
                        # primer: the plain decision prompt at this cut leaves a snapshot at exactly the prefix all probes
                        # of the group share (the server snapshots at the end of the last message), so probes do not each
                        # re-prefill. Same primer on every arm; its server-side time is recorded as kind=primer.
                        ptask = {**task, "id": "primer:%s:%d" % (task["traj"], task["cut_index"]), "kind": "primer"}
                        prec = run_task(ptask, trajs[task["traj"]], max_tokens=1)
                        out.write(json.dumps(prec) + "\n")
                        last_primer = key
                    rec = run_task(task, trajs[task["traj"]])
                except Exception as e:  # keep going; a dead server is restarted once per task
                    rec = {"id": task["id"], "traj": task["traj"], "kind": task["kind"], "error": repr(e)[:300]}
                    if p.poll() is not None:
                        restarts += 1
                        if restarts > 5:
                            raise
                        lf.close()
                        p, lf = start_server(arm, log_path)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if n % 20 == 0:
                    tm = rec.get("timings") or {}
                    print(n, len(todo), task["id"][-24:], rec.get("prompt_tokens"), tm, rec.get("error", ""), flush=True)
    finally:
        stop_server(p, lf)


if __name__ == "__main__":
    main()
