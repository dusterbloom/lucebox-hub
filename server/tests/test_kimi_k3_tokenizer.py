#!/usr/bin/env python3
"""Check the native tokenizer against frozen official Kimi-K3 token IDs."""

import argparse
import hashlib
import json
import struct
import subprocess


VECTORS = {
    "What's": [58434],
    "100": [1570],
    "1234567890": [6694, 12972, 16242, 15],
    "get_weather": [618, 21055, 2800],
    "can't": [58809],
    "fooBAR": [10570, 44894],
    "XMLHttpRequest": [19962, 9928, 3188],
    "中文abc": [16717, 8497],
    "newlines": [3514, 11541],
    "<|close|>think<|sep|>": [163588, 39964, 163589],
}

TOOL_PROMPT = (
    '<|open|>message role="system" type="tool-declare"<|sep|># Tools\n'
    'Here are the available tools, described in JSONSchema.\n\n```json\n'
    '[{"function":{"description":"Get current weather for a location",'
    '"name":"get_weather","parameters":{"properties":{"location":'
    '{"description":"City name","type":"string"}},"required":["location"],'
    '"type":"object"}},"type":"function"}]\n```<|close|>message<|sep|>'
    '<|end_of_msg|><|open|>message role="user"<|sep|>What\'s the weather in '
    'San Francisco?<|close|>message<|sep|><|end_of_msg|><|open|>message '
    'role="system" type="tool-choice"<|sep|>The system is invoked with '
    '`tool_choice=required`.\nYou MUST call tools in the next message.<|close|>'
    'message<|sep|><|end_of_msg|><|open|>message role="assistant"<|sep|>'
    '<|open|>response<|sep|>'
)
TOOL_PROMPT_SHA256 = "fa05c4fc98a796f387838fdef2aef89aaa5805ca2bbb1c7101ed428f80439d14"
TOOL_IDS_SHA256 = "f1ed3971af8259f3b9241d92404ec8d45f34137dec027fd96bc8d23b91b9773c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("harness")
    parser.add_argument("model")
    args = parser.parse_args()

    prompt_hash = hashlib.sha256(TOOL_PROMPT.encode()).hexdigest()
    assert prompt_hash == TOOL_PROMPT_SHA256, prompt_hash

    proc = subprocess.Popen(
        [args.harness, args.model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def encode(text: str) -> list[int]:
        proc.stdin.write(json.dumps({"cmd": "encode", "text": text}) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())["ids"]

    for text, expected in VECTORS.items():
        actual = encode(text)
        assert actual == expected, f"{text!r}: {actual} != {expected}"

    prompt_ids = encode(TOOL_PROMPT)
    packed = b"".join(struct.pack("<i", token) for token in prompt_ids)
    token_hash = hashlib.sha256(packed).hexdigest()
    assert len(prompt_ids) == 147, len(prompt_ids)
    assert token_hash == TOOL_IDS_SHA256, token_hash

    proc.stdin.write('{"cmd":"quit"}\n')
    proc.stdin.flush()
    assert proc.wait() == 0
    print("Kimi-K3 tokenizer oracle: PASS")


if __name__ == "__main__":
    main()
