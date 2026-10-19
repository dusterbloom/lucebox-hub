"""Decision-step task lists for the pin+UQ eval.
    python3 -I decision_steps.py select <R> <S> <out_dir>   eval list: warm-ups + S seeded steps with cut_tok > 16384 + R
    python3 -I decision_steps.py smoke <traj_index> <out_dir>  smoke list: one trajectory, warm-up + 2 steps (not eval rows)
A warm-up is the trajectory's last tool-result message boundary that fits 16K, sent with max_tokens 1 (kind=warmup, not scored), so the
system+tools block and the human question are prefilled densely, as in a real agent session."""
import json
import random
import sys

TASKS = "/mnt/kimi-k3/artifacts/agentic-eval/tasks_all60.jsonl"
TRAJ = "/mnt/kimi-k3/datasets/agentic-eval-prepared/trajectories.jsonl"
POOL, SEED = 16384, 20261019


def load():
    trajs = [json.loads(l) for l in open(TRAJ)]
    dec = [t for t in map(json.loads, open(TASKS)) if t["kind"] == "decision"]
    return trajs, dec


def warmup(tr):
    """Last message boundary <= POOL that ends with a tool message (boundaries[i] = tokens of messages[:i])."""
    i = max(int(k) for k, v in tr["boundaries"].items() if v <= POOL and tr["messages"][int(k) - 1]["role"] == "tool")
    return {"id": "warmup:%s:%d" % (tr["trajectory_id"], i), "traj": tr["trajectory_id"], "kind": "warmup",
            "cut_index": i, "cut_tok": tr["boundaries"][str(i)], "max_tokens": 1}


def plan(trajs, chosen):
    """Warm-up then the chosen steps in cut order, trajectories in file order."""
    out = []
    for tr in trajs:
        steps = sorted((t for t in chosen if t["traj"] == tr["trajectory_id"]), key=lambda t: t["cut_index"])
        if steps:
            out += [warmup(tr)] + steps
    return out


def write(out_dir, rows):
    with open(out_dir + "/decision_tasks.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out_dir + "/decision_ids.txt", "w") as f:
        f.writelines(r["id"] + "\n" for r in rows if r["kind"] == "decision")


def main():
    trajs, dec = load()
    if sys.argv[1] == "select":
        r, s, out_dir = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        elig = sorted((t for t in dec if t["cut_tok"] > POOL + r), key=lambda t: t["id"])
        chosen = random.Random(SEED).sample(elig, s)
        rows = plan(trajs, chosen)
        write(out_dir, rows)
        print(f"eligible {len(elig)} (cut_tok > {POOL + r}), chosen {s} over {len({t['traj'] for t in chosen})} trajectories, "
              f"{sum(r['kind'] == 'warmup' for r in rows)} warm-ups")
    else:
        tr, out_dir = trajs[int(sys.argv[2])], sys.argv[3]
        steps = sorted((t for t in dec if t["traj"] == tr["trajectory_id"] and t["cut_tok"] > POOL + 8192),
                       key=lambda t: t["cut_index"])[:2]
        rows = [{**t, "id": "smoke:" + t["id"]} for t in plan([tr], steps)]
        write(out_dir, rows)
        print("smoke", tr["trajectory_id"], [(r["id"], r["cut_tok"]) for r in rows])


if __name__ == "__main__":
    main()
