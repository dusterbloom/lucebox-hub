"""A1 core (PREREG "Rung A1" and Amendments 1 and 2): the shipping design on the B0 agentic traces, offline. Machine-agnostic:
no paths, no environment, no GPU assumptions beyond what the caller passes in. lucebox6 enters through lucebox/a1/a1_run.py,
Modal through modal/am_qwen35/a1.py; both call run_stage() below.

Per trajectory and arm: render the longest cut, tokenize once (the model's own tokenizer.json), run the arm over that stream
with one checkpoint at each distinct cut position (r2_core.gen_fullchk / gen_fblk, unchanged generators), and answer every
task of the cut from its checkpoint (r2_core.answer_ids). Arms and the row files (harness row schema, scored by the harness):
  FULL.jsonl   FULL-CHK, 640-token blocks, no compaction (the reference)
  SHIP.jsonl   F-BLK-UNI, every sealed 640-token block compacted once to 64 slots (ratio 0.10)
  FULLB.jsonl  FULL-CHK with 512-token blocks (reference sensitivity control)
The module imports torch and the compaction repo lazily, so the pure parts run anywhere.
"""
import hashlib
import json
import os
import sys
import time
from array import array
from dataclasses import dataclass, field

PROBE_MAX, DECISION_MAX = 64, 256            # run_arm.PROBE_MAX_TOKENS, DECISION_MAX_TOKENS (asserted against the harness)
BAR = 0.756                                   # B0 POOL16K retention, fixed by PREREG section 6
GATE3 = (0.35, 0.70)                          # reference yield after 5 trajectories (PREREG section 7)
EARLY_AT, EARLY_LEVEL = 20, 99.9              # Amendment 1: one Haybittle-Peto look at 20 trajectories, 99.9% interval
ARM_FILE = {"FULL": "FULL.jsonl", "SHIP": "SHIP.jsonl", "FULLB": "FULLB.jsonl"}
DISTANCE_BINS = [(0, 4096), (4096, 8192), (8192, 16384), (16384, 10 ** 9)]
MEM_USED_LIMIT_MIB, SWAP_LIMIT_MIB = 120 * 1024, 1024   # watchdog (PREREG section 3)


# ---------------------------------------------------------------- paths and harness

def default_paths(env=None, home=None):
    """Data and harness locations. $A1_TRAJ/$A1_TASKS/$A1_TEMPLATE/$A1_TOKENIZER/$A1_HARNESS override; otherwise lucebox6's
    ~/r1e/data/a1 + ~/r1e/code/agentic_eval when they exist, else the build host's locations."""
    env = os.environ if env is None else env
    home = home or os.path.expanduser("~")
    r1e_data, r1e_code = os.path.join(home, "r1e", "data", "a1"), os.path.join(home, "r1e", "code", "agentic_eval")
    here = os.path.dirname(os.path.abspath(__file__))
    repo_harness = os.path.normpath(os.path.join(here, "..", "..", "agentic_eval"))
    on_r1e = os.path.isdir(r1e_data)
    d = {"traj": r1e_data + "/trajectories.jsonl" if on_r1e else "/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl",
         "tasks": r1e_data + "/tasks_all60.jsonl" if on_r1e else "/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl",
         "template": r1e_data + "/qwen36_template.jinja" if on_r1e else "/mnt/kimi-k3/artifacts/kvflash-am-gpu/tmpl/qwen36_template.jinja",
         "tokenizer": r1e_data + "/qwen3.8-tokenizer.gguf" if on_r1e else "/mnt/kimi-k3/build/a1_render/qwen3.8-tokenizer.gguf",
         "harness": r1e_code if (on_r1e and os.path.isdir(r1e_code)) else repo_harness}
    for k, v in (("traj", "A1_TRAJ"), ("tasks", "A1_TASKS"), ("template", "A1_TEMPLATE"), ("tokenizer", "A1_TOKENIZER"),
                 ("harness", "A1_HARNESS")):
        if env.get(v):
            d[k] = env[v]
    return d


def file_hash(path, algo="md5"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Harness:
    """The B0 harness modules (convert, probes, run_arm; used unchanged, the template path is an attribute assignment) and the
    model's own tokenizer."""

    def __init__(self, paths):
        self.paths = paths
        if paths["harness"] not in sys.path:
            sys.path.insert(0, paths["harness"])
        import convert
        import probes
        import run_arm
        convert.TEMPLATE = paths["template"]
        assert (run_arm.PROBE_MAX_TOKENS, run_arm.DECISION_MAX_TOKENS) == (PROBE_MAX, DECISION_MAX)
        self.convert, self.probes, self.run_arm = convert, probes, run_arm
        if paths["tokenizer"].endswith(".gguf"):  # the server's own tokenizer (render parity, see server_tok.py)
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import server_tok
            self.tk = server_tok.ServerTokenizer(paths["tokenizer"])
            self.ids, self.decode = self.tk.encode, self.tk.decode
        else:  # HF tokenizer.json: 1 to 44 tokens short of the server on whitespace-only lines
            from tokenizers import Tokenizer
            self.tk = Tokenizer.from_file(paths["tokenizer"])
            self.tk.no_truncation()
            self.tk.no_padding()
            self.ids = lambda text: self.tk.encode(text).ids
            self.decode = lambda ids: self.tk.decode(list(ids), skip_special_tokens=False)

    def load_trajs(self):
        """{trajectory_id: record}, file order preserved."""
        out = {}
        with open(self.paths["traj"]) as f:
            for line in f:
                r = json.loads(line)
                out[r["trajectory_id"]] = r
        return out

    def load_tasks(self):
        with open(self.paths["tasks"]) as f:
            return [json.loads(l) for l in f]


def generated_text(H, history, gold, tools):
    """What the model generates for the assistant turn `gold` after `history`: the template's rendering of that turn behind
    the generation prompt, up to the stop token."""
    prompt = H.convert.render(history, tools, add_generation_prompt=True)
    full = H.convert.render(list(history) + [gold], tools)
    assert full.startswith(prompt), "assistant turn does not extend the generation prompt"
    text = full[len(prompt):]
    assert text.endswith("<|im_end|>\n")
    return text[: -len("<|im_end|>\n")]


# ---------------------------------------------------------------- plan: one stream per trajectory, checkpoints at the cuts

class PrefixError(AssertionError):
    pass


@dataclass
class Item:
    task: dict
    suffix: list      # token ids behind the cut prefix (question turn + generation prompt, or the generation prompt)
    max_new: int
    n_prompt: int     # tokens of the whole request
    sha: str          # request-id hash


@dataclass
class Cut:
    cut_index: int
    end: int          # checkpoint position = token length of the rendered cut prefix
    items: list = field(default_factory=list)


@dataclass
class Plan:
    traj: str
    tools: list
    stream: list      # token ids, built to the last cut
    cuts: list        # ascending by end


def ids_hash(ids):
    return hashlib.sha256(array("I", ids).tobytes()).hexdigest()


def plan_trajectory(H, traj, tasks):
    """Plan for `tasks` (all of one trajectory). Raises PrefixError unless every cut prefix is a token prefix of the stream and
    of every request built on it."""
    msgs, tools = traj["messages"], traj["tools"]
    cut_ids = {}
    for c in sorted({t["cut_index"] for t in tasks}):
        cut_ids[c] = H.ids(H.convert.render(msgs[:c], tools))
    if not cut_ids:
        return Plan(traj["trajectory_id"], tools, [], [])
    stream = cut_ids[max(cut_ids)]
    cuts = {c: Cut(c, len(p)) for c, p in cut_ids.items()}
    ends = [cuts[c].end for c in sorted(cuts)]
    if any(a >= b for a, b in zip(ends, ends[1:])):
        raise PrefixError(f"{traj['trajectory_id']}: cut positions not strictly increasing: {ends}")
    for c, p in cut_ids.items():
        if stream[: len(p)] != p:
            raise PrefixError(f"{traj['trajectory_id']}: cut {c} is not a prefix of the stream")
    for t in tasks:
        body = H.run_arm.build_request(t, traj)
        full = H.ids(H.convert.render(body["messages"], tools, add_generation_prompt=True))
        p = cut_ids[t["cut_index"]]
        if full[: len(p)] != p:
            raise PrefixError(f"{t['id']}: cut prefix is not a prefix of the request ids")
        cuts[t["cut_index"]].items.append(Item(t, full[len(p):], body["max_tokens"], len(full), ids_hash(full)))
    return Plan(traj["trajectory_id"], tools, stream, [cuts[c] for c in sorted(cuts)])


def restrict(plan, keep_ids):
    """Plan with only the items whose task id is in `keep_ids`; cuts left empty are dropped and the stream is cut at the last
    remaining checkpoint."""
    cuts = []
    for c in plan.cuts:
        items = [i for i in c.items if i.task["id"] in keep_ids]
        if items:
            cuts.append(Cut(c.cut_index, c.end, items))
    return Plan(plan.traj, plan.tools, plan.stream[: cuts[-1].end] if cuts else [], cuts)


def segs_from_checkpoints(stream, ends):
    """One r2_core.Seg per interval between consecutive checkpoints (the generators read only `ids`; `text` is unused)."""
    import torch
    import r2_core as r2
    assert list(ends) == sorted(set(ends)) and ends[-1] == len(stream)
    t = torch.tensor([stream], dtype=torch.long)
    return [r2.Seg("", t[:, a:b], range(0, b - a)) for a, b in zip([0] + list(ends[:-1]), ends)]


# ---------------------------------------------------------------- task selection (pure)

def pending(tasks, done_ids):
    return [t for t in tasks if t["id"] not in done_ids]


def subsample_decisions(tasks, stride):
    """Gate 2 (stage 2 only): decisions at positions `stride`, 2*stride, ... (1-based) in cut order; probes untouched."""
    if stride <= 1:
        return list(tasks)
    dec = sorted((t for t in tasks if t["kind"] == "decision"), key=lambda t: (t["cut_index"], t["id"]))
    keep = {t["id"] for t in dec[stride - 1:: stride]}
    return [t for t in tasks if t["kind"] != "decision" or t["id"] in keep]


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def retained_probe_ids(H, tasks, full_rows):
    """Probes the reference (FULL-CHK) answered correctly, scored by the harness."""
    R = {r["id"]: r for r in full_rows}
    return {t["id"] for t in tasks if t["kind"] == "probe" and t["id"] in R and "error" not in R[t["id"]]
            and H.probes.score_exact(R[t["id"]].get("content") or "", t["answer"])}


def parsed_decision_ids(H, tasks, full_rows):
    """Decision points where the reference produced a parsable first call (score.call_of)."""
    R = {r["id"]: r for r in full_rows}
    return {t["id"] for t in tasks if t["kind"] == "decision" and t["id"] in R and "error" not in R[t["id"]]
            and H.probes.parse_response_call({"tool_calls": R[t["id"]].get("tool_calls")}) is not None}


# ---------------------------------------------------------------- statistics (pure; r1b_stats resampling, other interval level)

def boot_ci(values, clusters, level=95.0, n_boot=10000, seed=0):
    """r1b_stats.boot_mean with a settable interval level: same cluster resampling and seed, percentile (100-level)/2 and
    100-(100-level)/2. At level 95 it equals boot_mean exactly (tested)."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    m = s[idx].sum(1) / n[idx].sum(1)
    tail = (100.0 - level) / 2
    lo, hi = np.percentile(m, [tail, 100.0 - tail])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n_q": int(len(values)), "n_art": int(len(uniq)),
            "level": level}


def early_verdict(ci, bar=BAR):
    """Haybittle-Peto look: PASS when the 99.9% lower bound is above the bar, FAIL when the upper bound is below it."""
    return "PASS" if ci["lo"] > bar else ("FAIL" if ci["hi"] < bar else "CONTINUE")


def retention_vectors(H, tasks, full_rows, ship_rows, trajs=None):
    """(values, clusters, task ids) over probes the reference got right that the shipping arm also answered; value = shipping
    arm correct. `trajs` restricts to those trajectory ids."""
    ret = retained_probe_ids(H, tasks, full_rows)
    S = {r["id"]: r for r in ship_rows}
    vals, cl, ids = [], [], []
    for t in tasks:
        if t["id"] in ret and t["id"] in S and (trajs is None or t["traj"] in trajs):
            vals.append(int("error" not in S[t["id"]] and H.probes.score_exact(S[t["id"]].get("content") or "", t["answer"])))
            cl.append(t["traj"])
            ids.append(t["id"])
    return vals, cl, ids


def reference_yield(H, tasks, full_rows, trajs=None):
    R = {r["id"]: r for r in full_rows}
    att = [t for t in tasks if t["kind"] == "probe" and t["id"] in R and (trajs is None or t["traj"] in trajs)]
    ok = sum(1 for t in att if "error" not in R[t["id"]] and H.probes.score_exact(R[t["id"]].get("content") or "", t["answer"]))
    return ok, len(att)


def retention_by_distance(tasks, ids, vals):
    by = {t["id"]: t for t in tasks}
    out = {}
    for lo, hi in DISTANCE_BINS:
        v = [x for i, x in zip(ids, vals) if lo <= by[i]["distance_tok"] < hi]
        out[f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"] = [sum(v), len(v)]
    return out


def stage_checks(H, tasks, full_rows, ship_rows, done_trajs, gate3_after=5, early_at=EARLY_AT):
    """Gate 3 and the early look on the trajectories finished so far (file order). Returns a dict with optional keys
    'gate3' (yield, ok, n, in_range) and 'early' (ci, verdict). Pure over the rows."""
    out = {}
    if len(done_trajs) >= gate3_after:
        first = set(done_trajs[:gate3_after])
        ok, n = reference_yield(H, tasks, full_rows, first)
        y = ok / n if n else 0.0
        out["gate3"] = {"trajectories": gate3_after, "correct": ok, "attempted": n, "yield": y,
                        "in_range": GATE3[0] <= y <= GATE3[1]}
    if len(done_trajs) >= early_at:
        first = set(done_trajs[:early_at])
        vals, cl, _ = retention_vectors(H, tasks, full_rows, ship_rows, first)
        if vals:
            ci = boot_ci(vals, cl, EARLY_LEVEL)
            out["early"] = {"ci": ci, "verdict": early_verdict(ci), "trajectories": early_at, "bar": BAR}
    return out


# ---------------------------------------------------------------- memory: gates and watchdog (pure parsers)

def parse_free(text):
    """`free -m` output -> {used, available, swap_used} in MiB."""
    out = {}
    for line in text.splitlines():
        f = line.split()
        if f and f[0] == "Mem:":
            out["used"], out["available"] = int(f[2]), int(f[6])
        elif f and f[0] == "Swap:":
            out["swap_used"] = int(f[2])
    return out


def watchdog_reason(mem, used_limit=MEM_USED_LIMIT_MIB, swap_limit=SWAP_LIMIT_MIB):
    if mem["used"] >= used_limit:
        return f"used {mem['used']} MiB >= {used_limit} MiB"
    if mem["swap_used"] > swap_limit:
        return f"swap used {mem['swap_used']} MiB > {swap_limit} MiB"
    return None


def launch_ok(mem, used_max=70 * 1024, avail_min=55 * 1024):
    """Pre-launch gate (PREREG section 3): used at most 70 GiB and available at least 55 GiB."""
    return mem["used"] <= used_max and mem["available"] >= avail_min


def memory_fraction(cap_gib, total_bytes):
    return min(1.0, cap_gib * 2 ** 30 / total_bytes)


# ---------------------------------------------------------------- rows

def strip_stop(ids, stop):
    ids = list(ids)
    while ids and ids[-1] in stop:
        ids.pop()
    return ids


def make_row(item, arm, gen_ids, text, parsed, stop, extra=None):
    """One row in the harness row schema (agentic_eval/run_arm.run_task) plus A1 fields."""
    t = item.task
    calls = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
             for c in parsed["tool_calls"]]
    ended = bool(gen_ids) and gen_ids[-1] in stop
    finish = "tool_calls" if calls else ("stop" if ended else "length")
    return {"id": t["id"], "traj": t["traj"], "kind": t["kind"], "content": parsed["cleaned_text"], "reasoning": "",
            "tool_calls": calls or None, "finish": finish, "prompt_tokens": item.n_prompt,
            "completion_tokens": len(gen_ids), "cached_tokens": 0, "client_wall_s": None,
            "arm": arm, "text": text, "gen_ids": list(gen_ids), "req_sha": item.sha, **(extra or {})}


def append_row(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# ---------------------------------------------------------------- running one unit (trajectory x arm)

def mem_gib():
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30


def run_unit(S, R, plan, gen, parse, emit, arm, log=print, det_repeat=0, empty_cache=True, **gen_kwargs):
    """Run `gen` (r2_core.gen_fullchk or gen_fblk) over plan.stream with a checkpoint at every cut, answering each item from its
    checkpoint. `parse(text, tools)` is the tool-parser program; `emit(row)` stores a row. Returns the unit's stats."""
    import torch
    import r2_core as r2
    if not plan.cuts:
        return {"traj": plan.traj, "arm": arm, "n_rows": 0}
    segs = segs_from_checkpoints(plan.stream, [c.end for c in plan.cuts])
    stop = S.eos
    stats = {"traj": plan.traj, "arm": arm, "n_rows": 0, "stream_tokens": len(plan.stream), "build_s": 0.0, "answer_s": 0.0,
             "checkpoints": []}
    t_unit = t_mark = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    n_det = det_repeat
    for k, triples, orig_len, info in gen(R, segs, None, list(range(len(segs))), **gen_kwargs):
        build_s = time.time() - t_mark
        cut = plan.cuts[k]
        assert orig_len == cut.end, (orig_len, cut.end)
        t_ans = time.time()
        for item in cut.items:
            tm = {}
            ids = r2.answer_ids(S, triples, orig_len, item.suffix, item.max_new, timing=tm)
            text = S.tok.decode(strip_stop(ids, stop), skip_special_tokens=False)
            extra = {"cut_index": cut.cut_index, "ctx_len": orig_len, "decode_tokens": len(ids),
                     "timings": {"prefill_ms": round(tm["prefill_s"] * 1000, 1), "decode_ms": round(tm["decode_s"] * 1000, 1),
                                 "prefilled_tokens": len(item.suffix), "cached_prefix_tokens": orig_len}}
            if n_det > 0:  # determinism check: the same item answered again from the same checkpoint
                n_det -= 1
                extra["repeat_equal"] = r2.answer_ids(S, triples, orig_len, item.suffix, item.max_new) == ids
            emit(make_row(item, arm, ids, text, parse(text, plan.tools), stop, extra))
            stats["n_rows"] += 1
        ans_s = time.time() - t_ans
        a, r_ = mem_gib()
        stats["build_s"] += build_s
        stats["answer_s"] += ans_s
        stats["checkpoints"].append({"k": k, "end": cut.end, "build_s": round(build_s, 2), "answer_s": round(ans_s, 2),
                                     "items": len(cut.items), "peak_alloc_gib": round(a, 1), "peak_reserved_gib": round(r_, 1)})
        log(f"[a1] {plan.traj[-8:]} {arm} cut {k + 1}/{len(segs)} ctx {orig_len} build {build_s:.1f}s answer {ans_s:.1f}s "
            f"({len(cut.items)} items) peak alloc {a:.1f} reserved {r_:.1f} GiB")
        del triples
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        t_mark = time.time()
    stats["wall_s"] = round(time.time() - t_unit, 1)
    stats["build_s"], stats["answer_s"] = round(stats["build_s"], 1), round(stats["answer_s"], 1)
    stats["peak_alloc_gib"], stats["peak_reserved_gib"] = (round(x, 1) for x in mem_gib())
    return stats


# ---------------------------------------------------------------- the stage loop

class StageStop(Exception):
    """A gate or the early look ended the stage; .reason says which."""

    def __init__(self, reason, detail=None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def run_stage(ctx, out, kinds=("probe",), arms=("FULL", "SHIP"), start=0, n=None, gate3_after=5, early_at=EARLY_AT,
              decision_stride=1, det_repeat=0, control_n=0, log=print):
    """Stage loop. ctx has S, R, H, parse. Trajectories in file order [start, start+n); per trajectory the arms in order; rows are
    appended per task, finished tasks are skipped on restart (resume per trajectory and arm). Returns a summary dict."""
    import r2_core as r2
    H, S, R = ctx.H, ctx.S, ctx.R
    os.makedirs(out, exist_ok=True)
    trajs = H.load_trajs()
    all_tasks = H.load_tasks()
    order = list(trajs)[start: None if n is None else start + n]
    by = {}
    for t in all_tasks:
        by.setdefault(t["traj"], []).append(t)
    paths = {a: os.path.join(out, f) for a, f in ARM_FILE.items()}
    progress = os.path.join(out, "progress.jsonl")
    finished = []

    def tasks_of(tid):
        return subsample_decisions([t for t in by.get(tid, []) if t["kind"] in kinds], decision_stride)

    def go(tid, arm, wanted_ids, gen, **kw):
        todo = {t["id"] for t in pending([t for t in tasks_of(tid) if t["id"] in wanted_ids],
                                         {r["id"] for r in read_rows(paths[arm])})}
        if not todo:
            return
        plan = restrict(plan_trajectory(H, trajs[tid], [t for t in tasks_of(tid) if t["id"] in todo]), todo)
        st = run_unit(S, R, plan, gen, ctx.parse, lambda row: append_row(paths[arm], row), arm, log=log,
                      det_repeat=det_repeat if tid == order[0] else 0, **kw)
        append_row(progress, st)

    summary = {"stopped": None}
    try:
        for tid in order:
            ids_all = {t["id"] for t in tasks_of(tid)}
            if "FULL" in arms:
                go(tid, "FULL", ids_all, r2.gen_fullchk, empty_cache=True)
            full_rows = [r for r in read_rows(paths["FULL"]) if r["traj"] == tid]
            keep = retained_probe_ids(H, tasks_of(tid), full_rows) | parsed_decision_ids(H, tasks_of(tid), full_rows)
            if "SHIP" in arms:
                go(tid, "SHIP", keep, r2.gen_fblk)
            finished.append(tid)
            done = set(finished)
            tasks_done = [t for t in all_tasks if t["traj"] in done]
            chk = stage_checks(H, tasks_done, read_rows(paths["FULL"]), read_rows(paths["SHIP"]), finished, gate3_after, early_at)
            with open(os.path.join(out, "checks.json"), "w") as f:
                json.dump(chk, f, indent=1)
            if "gate3" in chk and len(finished) == chk["gate3"]["trajectories"]:
                log(f"[a1] gate 3 after {len(finished)} trajectories: {chk['gate3']}")
                if not chk["gate3"]["in_range"]:
                    raise StageStop("gate3", chk["gate3"])
            if "early" in chk and len(finished) == early_at:
                log(f"[a1] early look at {early_at}: {chk['early']}")
                if chk["early"]["verdict"] != "CONTINUE":
                    raise StageStop("early-" + chk["early"]["verdict"], chk["early"])
    except StageStop as e:
        summary["stopped"] = {"reason": e.reason, "detail": e.detail}
        log(f"[a1] stage stopped: {e.reason}")
    if control_n:
        for tid in order[:control_n]:
            full_rows = [r for r in read_rows(paths["FULL"]) if r["traj"] == tid]
            go(tid, "FULLB", retained_probe_ids(H, tasks_of(tid), full_rows), r2.gen_fullchk, block=512, empty_cache=True)
    summary["trajectories"] = finished
    return summary


# ---------------------------------------------------------------- model loading (shared by every entry point)

def load_model(model_path, mem_cap_gib=None, log=print):
    """Qwen3.8-27B bf16 on the current CUDA/ROCm device, SDPA attention as R1(e). `mem_cap_gib` sets the per-process memory
    fraction (lucebox6: 100); None leaves it alone (Modal H200). Returns a SimpleNamespace(S, R, tok, model, gpu)."""
    from types import SimpleNamespace
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import r1a_core
    import r1b_core
    import r2_core as r2
    from models.qwen3_5 import HFGenerateShim
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda").eval()
    r1a_core.set_attn(model, "sdpa")
    if mem_cap_gib:
        torch.cuda.set_per_process_memory_fraction(memory_fraction(mem_cap_gib, torch.cuda.get_device_properties(0).total_memory), 0)
    gpu = torch.cuda.get_device_name(0)
    log(f"[a1] model loaded in {time.time() - t0:.0f}s on {gpu} torch {torch.__version__}")
    S = r1b_core.Session(model, tok, HFGenerateShim(model, tok, seed=1))
    return SimpleNamespace(S=S, R=r2.Runner(S, None, cell="FUNI"), tok=tok, model=model, gpu=gpu)
