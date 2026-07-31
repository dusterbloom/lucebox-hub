#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
mutation="$repo_root/formal/contracts/mutations/prefix-cache-bypass-selector.patch"

cd "$repo_root"
if [[ -n "$(git status --short --untracked-files=no)" ]]; then
    echo "mutation sensitivity requires a committed production checkout" >&2
    exit 2
fi

base_sha="$(git rev-parse HEAD)"
temporary_root="$(mktemp -d)"
trap 'rm -rf -- "$temporary_root"' EXIT
mutated_repo="$temporary_root/repo"
results="$temporary_root/results"

git clone --quiet --no-local "$repo_root" "$mutated_repo"
git -C "$mutated_repo" checkout --quiet --detach "$base_sha"
git -C "$mutated_repo" apply "$mutation"
git -C "$mutated_repo" config user.email "formal-mutation@example.invalid"
git -C "$mutated_repo" config user.name "Formal Mutation Test"
git -C "$mutated_repo" add server/src/server/prefix_cache_state.h
git -C "$mutated_repo" commit --quiet \
    -m "test: bypass production free-slot selector"

set +e
LUCEBOX_FORMAL_RESULTS="$results" \
    "$mutated_repo/scripts/formal.sh" --base-sha "$base_sha"
verification_status=$?
set -e

if [[ "$verification_status" -ne 10 ]]; then
    if [[ -f "$results/summary.md" ]]; then
        cat "$results/summary.md" >&2
    fi
    echo "expected counterexample exit 10, got $verification_status" >&2
    exit 1
fi

python3 -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
results = {item["id"]: item for item in report["results"]}
if report.get("conclusion") != "counterexample":
    raise SystemExit("mutation did not produce a counterexample conclusion")
if results["prefix-cache-inline"]["status"] != "verified":
    raise SystemExit("unrelated inline contract did not remain verified")
abort = results["prefix-cache-abort-hole"]
if abort["status"] != "counterexample":
    raise SystemExit("abort-hole target did not reject the call-site mutation")
native = abort.get("assumptions", {}).get("native_test", {})
if native.get("status") != "counterexample":
    raise SystemExit("base-approved native regression did not catch the mutation")
' "$results/report.json"

cat "$results/summary.md"
echo "abort-hole call-site mutation sensitivity: PASS"

full_mutated_repo="$temporary_root/full-lifecycle-repo"
full_results="$temporary_root/full-lifecycle-results"
full_mutation="$repo_root/formal/contracts/mutations/prefix-cache-full-bypass-free-slot.patch"

git clone --quiet --no-local "$repo_root" "$full_mutated_repo"
git -C "$full_mutated_repo" checkout --quiet --detach "$base_sha"
if ! git -C "$full_mutated_repo" apply --check "$full_mutation"; then
    echo "full lifecycle mutation patch no longer applies to exact HEAD" >&2
    exit 1
fi
git -C "$full_mutated_repo" apply "$full_mutation"
git -C "$full_mutated_repo" config user.email "formal-mutation@example.invalid"
git -C "$full_mutated_repo" config user.name "Formal Mutation Test"
git -C "$full_mutated_repo" add server/src/server/prefix_cache_state.h
git -C "$full_mutated_repo" commit --quiet \
    -m "test: bypass full prefix-cache free-slot selector"

full_native_binary="$temporary_root/test-full-prefix-cache-state"
c++ -std=c++17 -O0 -Wall -Wextra -Werror \
    -I "$full_mutated_repo/server/src" \
    "$full_mutated_repo/server/test/test_full_prefix_cache_state.cpp" \
    -o "$full_native_binary"
set +e
"$full_native_binary"
full_native_status=$?
set -e
if [[ "$full_native_status" -eq 0 ]]; then
    echo "full lifecycle native regression survived mutation" >&2
    exit 1
fi

set +e
LUCEBOX_FORMAL_RESULTS="$full_results" \
    "$full_mutated_repo/scripts/formal.sh" --base-sha "$base_sha"
full_verification_status=$?
set -e

if [[ "$full_verification_status" -ne 10 ]]; then
    if [[ -f "$full_results/summary.md" ]]; then
        cat "$full_results/summary.md" >&2
    fi
    echo "expected full lifecycle counterexample exit 10, got $full_verification_status" >&2
    exit 1
fi

python3 -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
results = {item["id"]: item for item in report["results"]}
if report.get("conclusion") != "counterexample":
    raise SystemExit("full lifecycle mutation did not produce a counterexample")
if results["prefix-cache-inline"]["status"] != "verified":
    raise SystemExit("unrelated inline contract did not remain verified")
full = results["prefix-cache-full-lifecycle"]
if full["status"] != "counterexample":
    raise SystemExit("full lifecycle target did not reject the mutation")
native = full.get("assumptions", {}).get("native_test", {})
if native.get("status") != "counterexample":
    raise SystemExit("full lifecycle native regression did not catch the mutation")
' "$full_results/report.json"

cat "$full_results/summary.md"
echo "full prefix-cache lifecycle mutation sensitivity: PASS"

# Keep the spec-commit mutation isolated from the prefix-cache clones and
# result sets so each target proves sensitivity to only its own defect.
spec_mutated_repo="$temporary_root/spec-commit/repo"
spec_results="$temporary_root/spec-commit/results"
spec_mutation="$repo_root/formal/contracts/mutations/spec-commit-invert-prefix-match.patch"
mkdir -p "$(dirname "$spec_mutated_repo")"

git clone --quiet --no-local "$repo_root" "$spec_mutated_repo"
git -C "$spec_mutated_repo" checkout --quiet --detach "$base_sha"
if ! git -C "$spec_mutated_repo" apply --check "$spec_mutation"; then
    echo "spec-commit-exactness: mutation patch no longer applies to exact HEAD" >&2
    exit 1
fi
git -C "$spec_mutated_repo" apply "$spec_mutation"
git -C "$spec_mutated_repo" config user.email "formal-mutation@example.invalid"
git -C "$spec_mutated_repo" config user.name "Formal Mutation Test"
git -C "$spec_mutated_repo" add server/src/common/spec_commit.h
git -C "$spec_mutated_repo" commit --quiet \
    -m "test: invert speculative prefix match"

spec_native_binary="$temporary_root/spec-commit/native-test"
c++ -std=c++17 -O0 -Wall -Wextra -Werror \
    -I "$spec_mutated_repo/server/src" \
    "$spec_mutated_repo/server/test/test_spec_commit.cpp" \
    -o "$spec_native_binary"
set +e
"$spec_native_binary"
spec_native_status=$?
set -e
if [[ "$spec_native_status" -eq 0 ]]; then
    echo "spec-commit-exactness: native regression survived mutation" >&2
    exit 1
fi

set +e
LUCEBOX_FORMAL_RESULTS="$spec_results" \
    "$spec_mutated_repo/scripts/formal.sh" --base-sha "$base_sha"
spec_verification_status=$?
set -e

if [[ "$spec_verification_status" -ne 10 ]]; then
    if [[ -f "$spec_results/summary.md" ]]; then
        cat "$spec_results/summary.md" >&2
    fi
    echo "spec-commit-exactness: expected counterexample exit 10, got $spec_verification_status" >&2
    exit 1
fi

python3 -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
results = {item["id"]: item for item in report["results"]}
if report.get("conclusion") != "counterexample":
    raise SystemExit("spec-commit mutation did not produce a counterexample")
spec = results["spec-commit-exactness"]
if spec["status"] != "counterexample":
    raise SystemExit("spec-commit target did not reject its mutation")
native = spec.get("assumptions", {}).get("native_test")
if native is not None and native.get("status") != "counterexample":
    raise SystemExit("spec-commit native result was not a counterexample")
' "$spec_results/report.json"

cat "$spec_results/summary.md"
echo "spec-commit exactness mutation sensitivity: PASS"
# Keep each additional mutation in its own clone so a failure cannot be
# masked by, or contaminate, another target's production edit.
# id|patch|mutated production path|base-native source
kvflash_mutation_cases=(
    "kvflash-residency-map|kvflash-capacity-off-by-one.patch|server/src/common/kvflash_residency_map.h|server/test/test_kvflash_residency_map.cpp"
)

for mutation_case in "${kvflash_mutation_cases[@]}"; do
    IFS='|' read -r case_id patch_name mutable_path native_source \
        <<< "$mutation_case"
    mutated_repo="$temporary_root/$case_id/repo"
    results="$temporary_root/$case_id/results"
    mutation="$repo_root/formal/contracts/mutations/$patch_name"
    mkdir -p "$(dirname "$mutated_repo")"

    git clone --quiet --no-local "$repo_root" "$mutated_repo"
    git -C "$mutated_repo" checkout --quiet --detach "$base_sha"
    if ! git -C "$mutated_repo" apply --check "$mutation"; then
        echo "$case_id: mutation patch no longer applies to exact HEAD" >&2
        exit 1
    fi
    git -C "$mutated_repo" apply "$mutation"
    git -C "$mutated_repo" config user.email "formal-mutation@example.invalid"
    git -C "$mutated_repo" config user.name "Formal Mutation Test"
    git -C "$mutated_repo" add "$mutable_path"
    git -C "$mutated_repo" commit --quiet \
        -m "test: inject $case_id mutation"

    native_binary="$temporary_root/$case_id/native-test"
    c++ -std=c++17 -O0 -Wall -Wextra -Werror \
        -I "$mutated_repo/server/src" \
        "$mutated_repo/$native_source" \
        -o "$native_binary"
    set +e
    "$native_binary"
    native_status=$?
    set -e
    if [[ "$native_status" -eq 0 ]]; then
        echo "$case_id: native regression survived mutation" >&2
        exit 1
    fi

    set +e
    LUCEBOX_FORMAL_RESULTS="$results" \
        "$mutated_repo/scripts/formal.sh" --base-sha "$base_sha"
    verification_status=$?
    set -e

    if [[ "$verification_status" -ne 10 ]]; then
        if [[ -f "$results/summary.md" ]]; then
            cat "$results/summary.md" >&2
        fi
        echo "$case_id: expected counterexample exit 10, got $verification_status" >&2
        exit 1
    fi

    python3 - "$results/report.json" "$case_id" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
target_id = sys.argv[2]
results = {item["id"]: item for item in report["results"]}
if report.get("conclusion") != "counterexample":
    raise SystemExit("mutation did not produce a counterexample conclusion")
if results[target_id]["status"] != "counterexample":
    raise SystemExit(f"{target_id} did not reject its mutation")
native = results[target_id].get("assumptions", {}).get("native_test")
if native is not None and native.get("status") != "counterexample":
    raise SystemExit(f"{target_id} native result was not a counterexample")
PY

    cat "$results/summary.md"
    echo "$case_id mutation sensitivity: PASS"
done
