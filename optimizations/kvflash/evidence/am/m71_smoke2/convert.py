"""OpenHands function-calling trajectories -> OpenAI-style messages, plus local rendering / token counting.

Pure functions only (no server, no GPU). Rendering uses the same jinja template the server is started with
(--chat-template-file), so token positions computed here match the server up to tojson formatting; the pilot
calibrates the residual against usage.prompt_tokens.
"""
import bisect
import json
import sys

USER_SITE = "/home/peppi/.local/lib/python3.12/site-packages"
if USER_SITE not in sys.path:  # python3 -I drops user site; add it explicitly (data readers run under -I)
    sys.path.append(USER_SITE)

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

TEMPLATE = "/mnt/kimi-k3/artifacts/kvflash-am-gpu/tmpl/qwen36_template.jinja"
TOKENIZER = "/mnt/kimi-k3/home-models/unsloth-Qwen3.6-27B-NVFP4/tokenizer.json"
# Every request (all arms) uses these, so the rendered history never changes when a user turn is appended.
CHAT_KWARGS = {"enable_thinking": False, "preserve_thinking": True}


def convert_messages(traj):
    """List of OpenHands dicts (content, name, role, tool_call_id, tool_calls) -> OpenAI messages.
    Returns None when a tool call has non-JSON arguments (trajectory is skipped, never repaired)."""
    out = []
    for m in traj:
        role = m["role"]
        if role in ("system", "user"):
            out.append({"role": role, "content": m["content"] or ""})
        elif role == "assistant":
            msg = {"role": "assistant", "content": m["content"] or ""}
            tcs = []
            for tc in m.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                try:
                    if not isinstance(json.loads(args), dict):
                        return None
                except (TypeError, ValueError):
                    return None
                tcs.append({"id": tc["id"], "type": "function",
                            "function": {"name": tc["function"]["name"], "arguments": args}})
            if tcs:
                msg["tool_calls"] = tcs
            out.append(msg)
        elif role == "tool":
            msg = {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"] or ""}
            if m.get("name"):
                msg["name"] = m["name"]
            out.append(msg)
        else:
            return None
    return out


def convert_tools(tools):
    """Drop None-valued keys that parquet struct unification adds, keep the OpenAI tool schema."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if v is not None}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return [clean(t) for t in tools]


_env = None


def _renderer():
    global _env
    if _env is None:
        env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
        env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)

        def raise_exception(msg):
            raise jinja2.TemplateError(msg)
        env.globals["raise_exception"] = raise_exception
        with open(TEMPLATE) as f:
            _env = env.from_string(f.read())
    return _env


def render(messages, tools, add_generation_prompt=False, **kw):
    """Render like the server's jinja path. arguments are parsed to dicts first, as the template expects mappings."""
    ms = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])}}
                               for tc in m["tool_calls"]]
        ms.append(m)
    kwargs = dict(CHAT_KWARGS)
    kwargs.update(kw)
    return _renderer().render(messages=ms, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)


_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_file(TOKENIZER)
        _tok.no_truncation()
        _tok.no_padding()
    return _tok


def count_tokens(text):
    return len(tokenizer().encode(text).ids)


def boundary_positions(messages, tools):
    """Token position (rendered prompt length without generation prompt) before each assistant message.
    Returns (full_tokens, {assistant_index: tokens_before}). The render of messages[:j] for an assistant index j
    is a character prefix of the full render, so one tokenization of the full text gives every boundary."""
    full = render(messages, tools)
    enc = tokenizer().encode(full)
    ends = [e for (_, e) in enc.offsets]
    pos = {}
    for j, m in enumerate(messages):
        if m["role"] == "assistant":
            n_chars = len(render(messages[:j], tools))
            assert full.startswith(render(messages[:j], tools)), "prefix property violated at %d" % j
            pos[j] = bisect.bisect_right(ends, n_chars)
    return len(enc.ids), pos


def qwen_tool_call_blocks(rendered):
    """Names of <tool_call><function=NAME> blocks in a rendered prompt (Qwen XML format)."""
    import re
    return re.findall(r"<tool_call>\n<function=([^>\n]+)>\n", rendered)
