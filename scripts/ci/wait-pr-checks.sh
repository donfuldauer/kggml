#!/usr/bin/env bash
# Wait, in the agent's own turn, until every check on a PR's head reaches a terminal state.
#
# PR activity subscriptions only wake an agent for FAILED checks, comments, reviews and the merge/close:
# a check that turns green produces no event, and in-session timers (cron / send_later) can vanish with
# the session. So an agent never ends its turn while checks are pending; it runs this script instead
# (Claude Code: Bash with run_in_background=true, whose exit re-invokes the agent; other harnesses: the
# foreground or their tracked-background equivalent). See AGENTS.md "Waiting for CI".
#
# Usage: scripts/ci/wait-pr-checks.sh <PR> [--poll SECONDS] [--queued-deadline MIN]
#                                          [--running-deadline MIN] [--no-fail-fast]
#
# Exit codes:
#   0  every check finished green (pass / skipped)
#   1  a check failed or was cancelled (with fail-fast, as soon as the first one does); a cancelled copy of a job
#      that also passed in another run on the same head (a duplicate) does not count
#   2  deadline hit: a check stayed queued past --queued-deadline after this script started (run it
#      right after the push) or ran past --running-deadline since it started = infrastructure failure
#   3  cannot query checks (no `gh`, not authenticated, PR not found): poll the check runs another way
#      (e.g. GitHub MCP `pull_request_read` method `get_check_runs`) on a short tracked timer instead
set -euo pipefail

usage() { sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 3; }

[[ $# -ge 1 ]] || usage
PR="$1"; shift
POLL=60
QUEUED_DEADLINE=20
RUNNING_DEADLINE=60
FAIL_FAST=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --poll) POLL="$2"; shift 2 ;;
    --queued-deadline) QUEUED_DEADLINE="$2"; shift 2 ;;
    --running-deadline) RUNNING_DEADLINE="$2"; shift 2 ;;
    --no-fail-fast) FAIL_FAST=0; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "wait-pr-checks: gh not found; poll the PR's check runs another way (exit 3)" >&2
  exit 3
fi

START_EPOCH="$(date +%s)"
ERR="$(mktemp)"
trap 'rm -f "$ERR"' EXIT

# collect_checks: prints the PR head's checks as JSON [{name, bucket, state, startedAt, link}], exit 3 if they cannot
# be read. GitHub Actions jobs come from the Actions API for every run on the head commit, not from `gh pr checks`:
# that keeps only the newest check per name, so when two runs share a job name (a `labeled` run next to the push's
# run, a re-run, a duplicate) a skipped copy in one run can hide the same job still running in the other. A job name
# is decided by its non-skipped copies: pending while any runs, failed if any failed, and a cancelled copy only counts
# when no other copy passed (a superseded or hand-cancelled duplicate does not fail the PR). Checks from apps that are
# not Actions jobs are taken from `gh pr checks` as they are.
collect_checks() {
  PR="$PR" python3 - <<'PY'
import json, os, subprocess, sys

def gh(*args, ok=(0,)):
    p = subprocess.run(["gh", *args], capture_output=True, text=True)
    if p.returncode not in ok:
        if "no checks reported" in p.stderr.lower():
            return ""
        sys.stderr.write(p.stderr)
        sys.exit(3)
    return p.stdout

pr = json.loads(gh("pr", "view", os.environ["PR"], "--json", "headRefOid,url"))
sha, url = pr["headRefOid"], pr["url"]
repo = "/".join(url.split("/")[3:5])

# gh exits 8 while checks are pending.
other = json.loads(gh("pr", "checks", os.environ["PR"], "--json", "name,bucket,state,startedAt,link", ok=(0, 8)) or "[]")
checks = [c for c in other if "/actions/runs/" not in (c.get("link") or "")]

runs = []
for page in range(1, 11):
    batch = json.loads(gh("api", f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100&page={page}"))["workflow_runs"]
    runs += batch
    if len(batch) < 100:
        break

def bucket(job):
    if job["status"] != "completed":
        return "pending"
    return {"success": "pass", "neutral": "pass", "skipped": "skipping", "cancelled": "cancel"}.get(job["conclusion"], "fail")

copies = {}
for run in runs:
    jobs = json.loads(gh("api", f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100"))["jobs"]
    for job in jobs:
        copies.setdefault((run["name"], job["name"]), []).append({
            "name": job["name"],
            "bucket": bucket(job),
            "state": (job["conclusion"] or job["status"] or "").upper(),
            "startedAt": job.get("started_at") or "",
            "link": job.get("html_url") or "",
        })

for (_, name), cs in sorted(copies.items()):
    live = [c for c in cs if c["bucket"] != "skipping"] or cs
    if any(c["bucket"] == "pending" for c in live):
        pick = [c for c in live if c["bucket"] == "pending"]
    elif any(c["bucket"] == "fail" for c in live):
        pick = [c for c in live if c["bucket"] == "fail"]
    elif any(c["bucket"] == "pass" for c in live):
        pick = [c for c in live if c["bucket"] == "pass"]
    else:
        pick = live
    checks.append(pick[0])
print(json.dumps(checks))
PY
}

while :; do
  set +e
  json="$(collect_checks 2>"$ERR")"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    cat "$ERR" >&2 || true
    echo "wait-pr-checks: cannot query checks for PR $PR (exit 3)" >&2
    exit 3
  fi

  set +e
  verdict="$(CHECKS_JSON="$json" START_EPOCH="$START_EPOCH" QUEUED_DEADLINE="$QUEUED_DEADLINE" \
    RUNNING_DEADLINE="$RUNNING_DEADLINE" FAIL_FAST="$FAIL_FAST" python3 - <<'PY'
import json, os, sys, time
from datetime import datetime

checks = json.loads(os.environ["CHECKS_JSON"] or "[]")
now = time.time()
waited_min = (now - int(os.environ["START_EPOCH"])) / 60
queued_deadline = float(os.environ["QUEUED_DEADLINE"])
running_deadline = float(os.environ["RUNNING_DEADLINE"])
fail_fast = os.environ["FAIL_FAST"] == "1"

def started_min(c):
    s = c.get("startedAt") or ""
    if not s or s.startswith("0001-"):
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    return (now - t) / 60

pending, failed, stale = [], [], []
for c in checks:
    bucket = c.get("bucket", "")
    label = f'{c.get("name")} [{c.get("state")}] {c.get("link", "")}'
    if bucket in ("fail", "cancel"):
        failed.append(label)
    elif bucket == "pending":
        pending.append(label)
        state = (c.get("state") or "").upper()
        run_min = started_min(c)
        if state in ("QUEUED", "PENDING", "WAITING", "REQUESTED", "EXPECTED") or run_min is None:
            if waited_min >= queued_deadline:
                stale.append(f"queued >= {queued_deadline:.0f} min: {label}")
        elif run_min >= running_deadline:
            stale.append(f"running {run_min:.0f} min >= {running_deadline:.0f} min: {label}")

stamp = time.strftime("%H:%M:%S")
print(f"[{stamp}] {len(checks)} checks: {len(pending)} pending, {len(failed)} failed", file=sys.stderr)
if failed and (fail_fast or not pending):
    print("FAILED\n" + "\n".join(failed)); sys.exit(1)
if stale:
    print("DEADLINE\n" + "\n".join(stale)); sys.exit(2)
if not pending and checks:
    print("GREEN"); sys.exit(0)
if not checks and waited_min >= queued_deadline:
    print(f"DEADLINE\nno checks reported {queued_deadline:.0f} min after start"); sys.exit(2)
sys.exit(10)  # keep waiting
PY
  )"
  code=$?
  set -e
  if [[ $code -ne 10 ]]; then
    echo "$verdict"
    exit "$code"
  fi
  sleep "$POLL"
done
