#!/usr/bin/env python3
"""One teacher-forced request for the LUCE_DECODE_REQ quality gate: the slice text goes in as the *assistant* turn
(a short user instruction before it), so the forced continuation is the model's own turn and the next-token
distribution is not dominated by the end-of-user-turn token, as it is when the text sits in the user turn.
The server (LUCE_TEACHER_TAIL / DROP / LOGITS) forces the last TAIL tokens of the rendered prompt minus the
DROP trailing template tokens. usage: teacher_client.py <port> <slice_file> <max_tokens>"""
import json, sys, urllib.request

port, path, max_tokens = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with open(path, encoding="utf-8") as f:
    text = f.read()
body = {"model": "bench", "temperature": 0, "max_tokens": max_tokens, "stream": False,
        "messages": [{"role": "user", "content": "Write the file."}, {"role": "assistant", "content": text}]}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1200) as resp:
    obj = json.load(resp)
print(json.dumps({"file": path, "usage": obj.get("usage"), "text": obj["choices"][0]["message"]["content"][:200]}))
