#!/usr/bin/env bash
# Regression test for scripts/ci/wait-pr-checks.sh against a fake `gh` (no network, no GitHub).
#
# Each case is a directory of poll fixtures: resp.<n> (the n-th poll, 0-based) or resp.last (every later poll), a
# JSON object {"runs": {"<run name>": [job, ...], ...}, "checks": [non-Actions check, ...]}. The fake answers
# `gh pr view` (and counts the poll), `gh pr checks`, and the Actions API calls for runs and jobs from it; a file
# resp.<n>.rc makes `gh pr view` fail with that exit code instead.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIT="$REPO_ROOT/scripts/ci/wait-pr-checks.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"
cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env python3
import json, os, sys
d = os.environ["STATE_DIR"]
counter = os.path.join(d, "n")
n = int(open(counter).read()) if os.path.exists(counter) else 0
args = sys.argv[1:]
if args[:2] == ["pr", "view"]:
    n += 1
    open(counter, "w").write(str(n))
poll = n - 1
f = os.path.join(d, f"resp.{poll}")
if not os.path.exists(f) and not os.path.exists(f + ".rc"):
    f = os.path.join(d, "resp.last")
if os.path.exists(f + ".rc"):
    sys.stderr.write("fake gh failure\n")
    sys.exit(int(open(f + ".rc").read()))
data = json.load(open(f))
runs = list(data.get("runs", {}).items())
if args[:2] == ["pr", "view"]:
    print(json.dumps({"headRefOid": "abc123", "url": "https://github.com/o/r/pull/1"}))
elif args[:2] == ["pr", "checks"]:
    print(json.dumps(data.get("checks", [])))
elif args[0] == "api" and "/jobs" in args[1]:
    run_id = int(args[1].split("/runs/")[1].split("/")[0])
    print(json.dumps({"jobs": runs[run_id][1]}))
elif args[0] == "api":
    page = int(args[1].split("page=")[-1])
    # Run names are stripped: fixtures tell two runs of one workflow apart by trailing spaces ("CI", "CI ").
    listed = [{"id": i, "name": name.strip()} for i, (name, _) in enumerate(runs)] if page == 1 else []
    print(json.dumps({"workflow_runs": listed}))
else:
    sys.exit(2)
GH
chmod +x "$T/bin/gh"

iso() { python3 -c "import sys,datetime;print((datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=int(sys.argv[1]))).strftime('%Y-%m-%dT%H:%M:%SZ'))" "$1"; }
NOW="$(iso 0)"; OLD="$(iso 61)"
# job <name> <status> <conclusion|null> <started_at|"">: an Actions job as the API returns it.
job() {
  local concl='null'; [[ "$3" == null ]] || concl="\"$3\""
  local started='null'; [[ -z "$4" ]] || started="\"$4\""
  printf '{"name":"%s","status":"%s","conclusion":%s,"started_at":%s,"html_url":"https://github.com/o/r/actions/runs/1/job/2"}' \
    "$1" "$2" "$concl" "$started"
}
pass() { job "$1" completed success "$NOW"; }
skip() { job "$1" completed skipped ""; }
# mk <case> <poll|last> <runs JSON object body> [gh exit code]
mk() { mkdir -p "$T/$1"; echo "{\"runs\": {$3}}" > "$T/$1/resp.$2"; [[ -z "${4:-}" ]] || echo "$4" > "$T/$1/resp.$2.rc"; }
fails=0
expect() { # name expected-exit args...
  local name="$1" want="$2"; shift 2
  set +e; STATE_DIR="$T/$name" PATH="$T/bin:$PATH" "$WAIT" 1 --poll 0 "$@" >/dev/null 2>&1; local got=$?; set -e
  if [[ $got -eq $want ]]; then echo "ok   $name (exit $got)"; else echo "FAIL $name: want $want got $got"; fails=$((fails+1)); fi
}

mk green last "\"CI\": [$(pass a),$(skip b)]"
expect green 0
mk failfast last "\"CI\": [$(job a in_progress null "$NOW"),$(job w completed failure "$NOW")]"
expect failfast 1
mk nofailfast 0 "\"CI\": [$(job a in_progress null "$NOW"),$(job w completed failure "$NOW")]"
mk nofailfast last "\"CI\": [$(pass a),$(job w completed failure "$NOW")]"
expect nofailfast 1 --no-fail-fast
mk cancelled last "\"CI\": [$(job a completed cancelled "$NOW")]"
expect cancelled 1
mk pending 0 "\"CI\": [$(job a queued null "")]"
mk pending 1 "\"CI\": [$(job a in_progress null "$NOW")]"
mk pending last "\"CI\": [$(pass a)]"
expect pending 0
mk queued last "\"CI\": [$(job a queued null "")]"
expect queued 2 --queued-deadline 0
mk running last "\"CI\": [$(job a in_progress null "$OLD")]"
expect running 2
mk ghfail last "" 1
expect ghfail 3
mkdir -p "$T/empty"
set +e; PATH="$T/empty" "$BASH" "$WAIT" 1 >/dev/null 2>&1; got=$?; set -e
if [[ $got -eq 3 ]]; then echo "ok   no-gh (exit $got)"; else echo "FAIL no-gh: want 3 got $got"; fails=$((fails+1)); fi

# Two runs on one head (a `labeled` run next to the push's run, or a re-run) sharing job names.
# A skipped copy in one run must not hide the same job still running in the other: wait for it.
mk hidden 0 "\"CI\": [$(pass ubuntu),$(job macos in_progress null "$NOW")], \"CI \": [$(skip ubuntu),$(skip macos)]"
mk hidden last "\"CI\": [$(pass ubuntu),$(job macos completed failure "$NOW")], \"CI \": [$(skip ubuntu),$(skip macos)]"
expect hidden 1
mk split last "\"CI\": [$(pass ubuntu),$(skip macos)], \"CI \": [$(skip ubuntu),$(pass macos)]"
expect split 0
# A hand-cancelled or superseded duplicate does not fail the PR when another copy passed; alone it does.
mk dupcancel last "\"CI\": [$(pass macos)], \"CI \": [$(job macos completed cancelled "$NOW")]"
expect dupcancel 0
mk onlycancel last "\"CI\": [$(job macos completed cancelled "$NOW")], \"CI \": [$(skip macos)]"
expect onlycancel 1
# A failure in either copy fails.
mk dupfail last "\"CI\": [$(pass macos)], \"CI \": [$(job macos completed failure "$NOW")]"
expect dupfail 1
# Checks that are not Actions jobs (apps) still count.
mkdir -p "$T/app"
echo "{\"runs\": {\"CI\": [$(pass a)]}, \"checks\": [{\"name\":\"app\",\"bucket\":\"fail\",\"state\":\"FAILURE\",\"startedAt\":\"$NOW\",\"link\":\"https://example.invalid/app\"}]}" > "$T/app/resp.last"
expect app 1

[[ $fails -eq 0 ]] && echo "wait-pr-checks: all tests passed" || { echo "wait-pr-checks: $fails failure(s)"; exit 1; }
