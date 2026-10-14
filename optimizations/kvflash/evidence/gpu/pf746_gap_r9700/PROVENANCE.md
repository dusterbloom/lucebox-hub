# Provenance

## Git
- Base refs: `kvflash/session-restore-main` = 30c099359d686dc2e3f058bc86c2e255e22286dd (2.0a side, `ModelBackend::restorable_prefix`),
  `refs/remotes/pr/746` = eca7265bd69856d6b97ffaada64989146a1c8eee (PR #746 head), merge base fbcefbc422fd38ac69c91197d4ef4a8426f2c46c.
- Arm A branch `kvflash/pf746-merge` = da9347580d91767430a03930f40221101fa17ef3 (parents 30c09935, eca7265b), made without a checkout:
  `git merge-tree --write-tree kvflash/session-restore-main refs/remotes/pr/746` (tree 31e9cd2c1a66c0972db318fa56478ea51bb970ae, clean),
  `git commit-tree 31e9cd2c... -p 30c09935... -p eca7265b... -m "merge: ..."`, `git branch kvflash/pf746-merge da934758...`.
- Arm B branch `kvflash/pf746-merge-fix` = 78b483bb8e763ec0a6d7765e4243c7db663df503 (parent da934758): one commit changing
  `server/src/server/http_server.cpp` `HttpServer::restorable_prefix_tokens` as specified (diff in `arm_B_fix.patch`), edited in the worktree
  `/home/peppi/Dev/lucebox-builds/wt-pf746-fix`. Commit messages carry no attribution lines.
- Pushed only to the lucebox6 bare repo (`git push lucebox6 <branch>:refs/heads/<branch>`, `/home/duster/lucebox.git`). Nothing pushed to GitHub.

## Build (lucebox6; `build.sh`, `build_A.log`, `build_B.log` in this directory)
- `~/pf746/build.sh kvflash/pf746-merge A` and `~/pf746/build.sh kvflash/pf746-merge-fix B`, clone `~/pf746/src` of the bare repo,
  cmake Release, `-DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1201 -DLUCE_HIP_SM80_EQUIV=ON` (the flags of `~/kvflash-gate/build.sh`), `TMPDIR=~/pf746/tmp`.
- `~/pf746/bin/luce_server-A` md5 39d0d160e05973ede56aa69dc94b71be (da934758); `~/pf746/bin/luce_server-B` md5 ab232f231e91cbf938191b9a5edd1c4e (78b483bb).
  Arm C reuses luce_server-B.

## Models (lucebox6 `~/models`)
- `Qwen3.8-27B-UD-Q4_K_M.gguf` md5 76f527022f010d42bc0dff5905c60444 (target, every arm).
- `Qwen3.5-0.8B-BF16.gguf` md5 16671eda429ac1aaff071a3f903380da (copied from `/mnt/kimi-k3/home-models/qwen3.5-0.8b/`, md5 verified on both ends).
- Data (read-only): `~/r1e/code/repo/data/QuALITY.v1.0.1.htmlstripped.dev`, article ids 22073, 22346, 22876, 22590, 22867.

## Run
- Host lucebox6, GPU0 = AMD Radeon R9700 gfx1201 (32 GB), `HIP_VISIBLE_DEVICES=0`; GPU1 and the r1e processes were not touched. Date 2026-10-14 (final run 15:07 to 15:12 UTC).
- Final run: `cd ~/pf746 && KV_POOL=16384 PASTE_IDS=22073,22346,22876,22590,22867 PASTE_CHARS=28000,28000,40000,40000,40000 scripts/run_all.sh att3_final`
  (arms A, B, C in turn, each through `run_arm.sh`, flock on `~/kvflash-gate/server.lock`).
- Exact server command lines and env per arm: `final/cmdline_A.json`, `final/cmdline_B.json`, `final/cmdline_C.json`; md5s and branch SHAs per arm: `final/provenance_<ARM>.txt`.
  All arms: `luce_server MODEL --host 127.0.0.1 --port 18292 --max-ctx 65536 --kvflash 16384 --model-name luce --prefill-drafter DRAFTER
  --prefill-compression {auto|auto|off} --prefill-threshold 16384 --prefill-skip-park --no-kv-cache`, env `PFLASH_ADMISSION=cache PFLASH_SELECT_MODE=budget_only
  PFLASH_SELECT_SCORER=legacy HIP_VISIBLE_DEVICES=0 TMPDIR=~/pf746/tmp`; every inherited `KVFLASH_*`, `PFLASH_*`, `LUCE_*` variable is removed first.
- Calibration attempts 1 and 2 used earlier revisions of `pf746_gap.py` (attempt 1: first-eligible article selection; attempt 2: explicit ids, disk KV cache still on); their
  `provenance_A.txt` and `cmdline_A.json` are in `attempts/`.
- Nothing of this work was written to /tmp on lucebox6 (checked: no new files; builds, logs, scripts and models are under `~/pf746`, `~/models`, `~/kvflash-gate`).
  Locally, the only /tmp use is the Claude Code harness's own output file for the backgrounded drafter copy (scp) command.
