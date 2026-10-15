"""Pure tests for a1_core (no model, no GPU): python3 -m pytest -q lucebox/a1/test_a1_core.py
Real-data tests are skipped when the A1 data is absent."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../modal/am_qwen35"))
import a1_core as core  # noqa: E402

PATHS = core.default_paths()
HAVE_DATA = all(os.path.exists(PATHS[k]) for k in ("traj", "tasks", "template", "tokenizer"))


def _item(tid, kind="probe", suffix=(1, 2), traj="T"):
    return core.Item({"id": tid, "traj": traj, "kind": kind}, list(suffix), 64, 100, "sha")


def _plan():
    cuts = [core.Cut(4, 10, [_item("a"), _item("b")]), core.Cut(8, 25, [_item("c")]), core.Cut(12, 40, [_item("d")])]
    return core.Plan("T", [], list(range(50)), cuts)


def test_default_paths_env_overrides(tmp_path):
    p = core.default_paths(env={"A1_TRAJ": "/x/t.jsonl", "A1_HARNESS": "/x/h"}, home=str(tmp_path))
    assert p["traj"] == "/x/t.jsonl" and p["harness"] == "/x/h" and p["tasks"].endswith("tasks_all60.jsonl")


def test_default_paths_prefers_r1e_layout(tmp_path):
    (tmp_path / "r1e" / "data" / "a1").mkdir(parents=True)
    (tmp_path / "r1e" / "code" / "agentic_eval").mkdir(parents=True)
    p = core.default_paths(env={}, home=str(tmp_path))
    assert p["traj"] == f"{tmp_path}/r1e/data/a1/trajectories.jsonl" and p["harness"] == f"{tmp_path}/r1e/code/agentic_eval"
    assert p["tokenizer"] == f"{tmp_path}/r1e/data/a1/qwen3.8-tokenizer.gguf"


def test_restrict_drops_items_empty_cuts_and_trailing_stream():
    r = core.restrict(_plan(), {"a", "c"})
    assert [(c.cut_index, [i.task["id"] for i in c.items]) for c in r.cuts] == [(4, ["a"]), (8, ["c"])]
    assert len(r.stream) == 25                       # built only to the last remaining checkpoint
    assert core.restrict(_plan(), set()).cuts == [] and core.restrict(_plan(), set()).stream == []
    assert len(core.restrict(_plan(), {"d"}).stream) == 40


def test_subsample_decisions_positions_2_4_6_in_cut_order():
    tasks = [{"id": f"d{i}", "kind": "decision", "cut_index": 10 - i} for i in range(6)] + [{"id": "p", "kind": "probe", "cut_index": 1}]
    kept = [t["id"] for t in core.subsample_decisions(tasks, 2)]
    assert sorted(kept) == ["d0", "d2", "d4", "p"]
    assert core.subsample_decisions(tasks, 1) == tasks


def test_pending_skips_done_ids():
    assert [t["id"] for t in core.pending([{"id": "a"}, {"id": "b"}], {"a"})] == ["b"]


def test_strip_stop_only_trailing():
    assert core.strip_stop([5, 7, 9, 9], {9}) == [5, 7] and core.strip_stop([9, 5], {9}) == [9, 5] and core.strip_stop([], {9}) == []


def test_ids_hash_is_order_and_value_sensitive():
    assert core.ids_hash([1, 2, 3]) == core.ids_hash([1, 2, 3]) != core.ids_hash([1, 3, 2])
    assert len(core.ids_hash([1])) == 64 and core.ids_hash([1, 2]) != core.ids_hash([1, 2, 0])


def test_boot_ci_matches_boot_mean_at_95():
    from r1b_stats import boot_mean
    vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    cl = list("aaabbbcccddd")
    a, b = core.boot_ci(vals, cl, 95.0), boot_mean(vals, cl)
    assert (a["mean"], a["lo"], a["hi"], a["n_q"], a["n_art"]) == (b["mean"], b["lo"], b["hi"], b["n_q"], b["n_art"])
    w = core.boot_ci(vals, cl, 99.9)
    assert w["lo"] <= a["lo"] and w["hi"] >= a["hi"]


def test_early_verdict_haybittle_peto():
    assert core.early_verdict({"lo": 0.80, "hi": 0.9}) == "PASS"
    assert core.early_verdict({"lo": 0.70, "hi": 0.74}) == "FAIL"
    assert core.early_verdict({"lo": 0.70, "hi": 0.80}) == "CONTINUE"
    assert core.early_verdict({"lo": 0.756, "hi": 0.9}) == "CONTINUE"     # strict inequalities


FREE = """               total        used        free      shared  buff/cache   available
Mem:          128089       66572       62100           0         501       61517
Swap:           8191           0        8191
"""


def test_parse_free_and_gates():
    m = core.parse_free(FREE)
    assert m == {"used": 66572, "available": 61517, "swap_used": 0}
    assert core.launch_ok(m) and not core.launch_ok({**m, "used": 71 * 1024}) and not core.launch_ok({**m, "available": 50 * 1024})
    assert core.watchdog_reason(m) is None
    assert "used" in core.watchdog_reason({**m, "used": 120 * 1024})
    assert core.watchdog_reason({**m, "used": 120 * 1024 - 1}) is None
    # swap already in use when the run starts is not the run's (lucebox6 idles at 459 MiB): the limit is on the growth
    assert core.watchdog_reason({**m, "swap_used": 1483}, swap_base=459) is None
    assert "swap" in core.watchdog_reason({**m, "swap_used": 1484}, swap_base=459)
    assert "swap" in core.watchdog_reason({**m, "swap_used": 1025}) and core.watchdog_reason({**m, "swap_used": 1024}) is None


def test_memory_fraction():
    assert core.memory_fraction(100, 125 * 2 ** 30) == pytest.approx(0.8)
    assert core.memory_fraction(100, 80 * 2 ** 30) == 1.0


@pytest.mark.skipif(not HAVE_DATA, reason="A1 data not present")
class TestWithHarness:
    @pytest.fixture(scope="class")
    def H(self):
        return core.Harness(PATHS)

    def test_make_row_is_scored_by_the_harness(self, H):
        parsed = {"cleaned_text": "ok", "tool_calls": [{"id": "c1", "name": "execute_bash", "arguments": '{"command": "ls -la"}'}]}
        row = core.make_row(_item("T:d1", "decision"), "FULL", [5, 6, 9], "x", parsed, {9}, {"ctx_len": 3})
        assert H.probes.parse_response_call({"tool_calls": row["tool_calls"]}) == ("execute_bash", "execute_bash|ls|")
        assert row["finish"] == "tool_calls" and row["completion_tokens"] == 3 and row["ctx_len"] == 3
        for k in ("id", "traj", "kind", "content", "tool_calls", "finish", "prompt_tokens", "completion_tokens", "cached_tokens"):
            assert k in row
        plain = core.make_row(_item("T:p1"), "FULL", [5, 6], "src/x.py", {"cleaned_text": "src/x.py", "tool_calls": []}, {9})
        assert plain["tool_calls"] is None and plain["finish"] == "length" and plain["content"] == "src/x.py"

    def test_retained_and_parsed_sets(self, H):
        tasks = [{"id": "p1", "traj": "T", "kind": "probe", "answer": "foo.py"}, {"id": "p2", "traj": "T", "kind": "probe", "answer": "bar"},
                 {"id": "d1", "traj": "T", "kind": "decision"}, {"id": "d2", "traj": "T", "kind": "decision"}]
        call = [{"id": "c", "type": "function", "function": {"name": "execute_bash", "arguments": '{"command": "ls"}'}}]
        rows = [{"id": "p1", "content": " `foo.py`. "}, {"id": "p2", "content": "baz"},
                {"id": "d1", "tool_calls": call}, {"id": "d2", "tool_calls": None, "content": "text"}]
        assert core.retained_probe_ids(H, tasks, rows) == {"p1"} and core.parsed_decision_ids(H, tasks, rows) == {"d1"}

    def test_stage_checks_gate3_and_early_look(self, H):
        def world(n_traj, ship_ok, full_ok=True):
            tasks, full, ship, done = [], [], [], []
            for i in range(n_traj):
                tid = f"T{i}"
                done.append(tid)
                for j in range(4):
                    t = {"id": f"{tid}:p{j}", "traj": tid, "kind": "probe", "answer": "a", "distance_tok": 5000}
                    tasks.append(t)
                    full.append({"id": t["id"], "content": "a" if full_ok else "b"})
                    ship.append({"id": t["id"], "content": "a" if ship_ok else "b"})
            return tasks, full, ship, done

        tasks, full, ship, done = world(20, True)
        c = core.stage_checks(H, tasks, full, ship, done)
        assert c["gate3"]["in_range"] is False and c["gate3"]["yield"] == 1.0          # 100% is outside 35-70%
        assert c["early"]["verdict"] == "PASS" and c["early"]["ci"]["level"] == 99.9
        assert core.stage_checks(H, *world(20, False)[:3], done)["early"]["verdict"] == "FAIL"
        c19 = core.stage_checks(H, *world(19, True))
        assert "early" not in c19 and "gate3" in c19
        assert core.stage_checks(H, *world(4, True)) == {}

    def test_retention_by_distance_bins(self, H):
        tasks = [{"id": f"p{i}", "distance_tok": d} for i, d in enumerate([0, 4095, 4096, 9000, 16384, 20000])]
        got = core.retention_by_distance(tasks, [t["id"] for t in tasks], [1, 0, 1, 1, 0, 1])
        assert got == {"0-4096": [1, 2], "4096-8192": [1, 1], "8192-16384": [1, 1], "16384-inf": [1, 2]}

    def test_plan_first_trajectory_probes(self, H):
        trajs = H.load_trajs()
        tid = next(iter(trajs))
        tasks = [t for t in H.load_tasks() if t["traj"] == tid and t["kind"] == "probe"]
        plan = core.plan_trajectory(H, trajs[tid], tasks)
        ends = [c.end for c in plan.cuts]
        assert ends == sorted(set(ends)) and ends[-1] == len(plan.stream)
        assert sum(len(c.items) for c in plan.cuts) == len(tasks)
        it = plan.cuts[0].items[0]
        assert H.decode(it.suffix).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert it.max_new == 64 and it.n_prompt == plan.cuts[0].end + len(it.suffix)

    def test_generated_text_strips_stop(self, H):
        m = [{"role": "user", "content": "q"},
             {"role": "assistant", "content": "hi", "tool_calls": [{"id": "c", "type": "function",
                                                                    "function": {"name": "t", "arguments": '{"a": "b"}'}}]}]
        txt = core.generated_text(H, m[:1], m[1], [])
        assert txt.startswith("hi") and "<tool_call>" in txt and "<|im_end|>" not in txt
