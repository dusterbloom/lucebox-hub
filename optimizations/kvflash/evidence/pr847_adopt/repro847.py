"""#847 scenario: a dense snapshot written without --kvflash, found by a --kvflash 8192 server.
usage (python3 -I): KV_ROOT KV_BIN KV_MODEL TMPDIR KV_MAX_CTX KV_PORT set; argv[1] = arm name."""
import sys, os, json, shutil, urllib.request, urllib.error
sys.path.insert(0, "/home/peppi/Dev/lucebox-builds/wt-m7/optimizations/kvflash/am_gate")
from kvlib import Server, filler, ROOT, URL

Q = "\nQuestion: continue this list with the next 12 records in exactly the same format and write nothing else."
P = filler(12500, salt=31) + Q


def req(messages, max_tokens=32):
    body = json.dumps({"model": "luce", "messages": messages, "temperature": 0, "seed": 1,
                       "max_tokens": max_tokens}).encode()
    r = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=3600) as f:
            j = json.loads(f.read())
            return {"status": f.status, "text": j["choices"][0]["message"]["content"],
                    "prompt_tokens": j["usage"]["prompt_tokens"]}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "error": e.read().decode(errors="replace")[:400]}
    except Exception as e:
        return {"status": None, "error": f"{type(e).__name__}: {e}"}


arm = sys.argv[1]
d = f"{ROOT}/kvdir"
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)
out = {"arm": arm, "bin": os.environ["KV_BIN"]}
user = [{"role": "user", "content": P}]
with Server(f"{ROOT}/a_writer.log", extra=["--kv-cache-dir", d]):
    out["a_writer"] = req(user)
out["dkv_files"] = [f for _, _, fs in os.walk(d) for f in fs if f.endswith(".dkv")]
with Server(f"{ROOT}/b_kvflash8192.log", extra=["--kvflash", "8192", "--kv-cache-dir", d]):
    out["b_same_request"] = req(user)
    follow = user + [{"role": "assistant", "content": out["b_same_request"].get("text", "ok")},
                     {"role": "user", "content": "Thanks. Now give just the last record of your list."}]
    out["b_followup"] = req(follow)
d2 = f"{ROOT}/kvdir_empty"
shutil.rmtree(d2, ignore_errors=True)
os.makedirs(d2)
with Server(f"{ROOT}/c_reference.log", extra=["--kvflash", "8192", "--kv-cache-dir", d2]):
    out["c_empty_cache_reference"] = req(user)
b, c = out["b_same_request"], out["c_empty_cache_reference"]
out["first_answer_equals_reference"] = bool(b.get("text")) and b.get("text") == c.get("text")
json.dump(out, open(f"{ROOT}/result.json", "w"), indent=1)
print(json.dumps(out, indent=1))
