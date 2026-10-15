"""Request-id hashes for A1 (PREREG section 2, CPU tests): the token ids of every B0 request (the harness render of
build_request, tokenized with the model's own tokenizer.json) and of every cut prefix, hashed with sha256 over little-endian
uint32. Computed on the build host and again on lucebox6; the files must be identical.

  python3 a1_hash.py write <file>      # compute and write
  python3 a1_hash.py check <file>      # compute and compare with <file>; exit 1 on any difference
plan_trajectory() raises PrefixError if any cut prefix is not a token prefix of the stream or of a request built on it.
"""
import hashlib
import sys
import time

import a1_core as core


def request_hashes(H, log=print):
    """([(task id, n_tokens, sha)] in task-file order, [('traj:cut_index', n_tokens, sha)] sorted, n trajectories)."""
    trajs = H.load_trajs()
    by = {}
    for t in H.load_tasks():
        by.setdefault(t["traj"], []).append(t)
    reqs, cuts = {}, []
    t0 = time.time()
    for n, (tid, tasks) in enumerate(by.items(), 1):
        plan = core.plan_trajectory(H, trajs[tid], tasks)
        for c in plan.cuts:
            cuts.append((f"{tid}:{c.cut_index}", c.end, core.ids_hash(plan.stream[: c.end])))
            for it in c.items:
                reqs[it.task["id"]] = (it.task["id"], it.n_prompt, it.sha)
        log(f"[hash] {n}/{len(by)} {tid} {len(plan.cuts)} cuts {time.time() - t0:.0f}s")
    order = [t["id"] for ts in by.values() for t in ts]
    return [reqs[i] for i in order], sorted(cuts), len(by)


def hash_file_lines(H, reqs, cuts):
    p = H.paths
    allr = hashlib.sha256("".join(f"{a}\t{b}\t{c}\n" for a, b, c in reqs).encode()).hexdigest()
    allc = hashlib.sha256("".join(f"{a}\t{b}\t{c}\n" for a, b, c in cuts).encode()).hexdigest()
    head = [f"# A1 request-id hashes: sha256 of the uint32 token ids; render = harness convert.render of run_arm.build_request, "
            f"tokenizer = the model's own tokenizer.json",
            f"# tokenizer_sha256\t{core.file_hash(p['tokenizer'], 'sha256')}",
            f"# template_md5\t{core.file_hash(p['template'])}",
            f"# trajectories_md5\t{core.file_hash(p['traj'])}",
            f"# tasks_md5\t{core.file_hash(p['tasks'])}"]
    return (head + [f"req:{a}\t{b}\t{c}" for a, b, c in reqs] + [f"ALL_REQUESTS\t{len(reqs)}\t{allr}"]
            + [f"cut:{a}\t{b}\t{c}" for a, b, c in cuts] + [f"ALL_CUTS\t{len(cuts)}\t{allc}"])


def read_hash_file(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def main(argv):
    mode, path = argv[1], argv[2]
    H = core.Harness(core.default_paths())
    lines = hash_file_lines(H, *request_hashes(H)[:2])
    if mode == "write":
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} lines to {path}; {lines[len(lines) - 1]}; "
              f"{next(l for l in lines if l.startswith('ALL_REQUESTS'))}")
        return 0
    want = read_hash_file(path)
    diff = [(a, b) for a, b in zip(lines, want) if a != b]
    ok = lines == want
    print(("IDENTICAL" if ok else f"DIFFERENT ({len(diff)} lines, first: {diff[:1]})"), next(l for l in lines if l.startswith("ALL_")))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
