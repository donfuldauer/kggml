#!/usr/bin/env bash
# Regression test for scripts/ci/wait-pr-checks.sh against a fake `gh` (no network, no GitHub).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIT="$REPO_ROOT/scripts/ci/wait-pr-checks.sh"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin"
cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env bash
n=$(cat "$STATE_DIR/n" 2>/dev/null || echo 0); echo $((n+1)) > "$STATE_DIR/n"
f="$STATE_DIR/resp.$n"; [[ -f $f ]] || f="$STATE_DIR/resp.last"
cat "$f"; exit "$(cat "$f.rc" 2>/dev/null || echo 0)"
GH
chmod +x "$T/bin/gh"

iso() { python3 -c "import sys,datetime;print((datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=int(sys.argv[1]))).strftime('%Y-%m-%dT%H:%M:%SZ'))" "$1"; }
NOW="$(iso 0)"; OLD="$(iso 61)"
chk() { printf '{"name":"%s","bucket":"%s","state":"%s","startedAt":"%s","link":"l"}' "$@"; }
mk() { mkdir -p "$T/$1"; echo "[$3]" > "$T/$1/resp.$2"; [[ -z "${4:-}" ]] || echo "$4" > "$T/$1/resp.$2.rc"; }
fails=0
expect() { # name expected-exit args...
  local name="$1" want="$2"; shift 2
  set +e; STATE_DIR="$T/$name" PATH="$T/bin:$PATH" "$WAIT" 1 --poll 0 "$@" >/dev/null 2>&1; local got=$?; set -e
  if [[ $got -eq $want ]]; then echo "ok   $name (exit $got)"; else echo "FAIL $name: want $want got $got"; fails=$((fails+1)); fi
}

mk green last "$(chk a pass SUCCESS "$NOW"),$(chk b skipping SKIPPED "")"
expect green 0
mk failfast last "$(chk a pending IN_PROGRESS "$NOW"),$(chk w fail FAILURE "$NOW")" 8
expect failfast 1
mk nofailfast 0 "$(chk a pending IN_PROGRESS "$NOW"),$(chk w fail FAILURE "$NOW")" 8
mk nofailfast last "$(chk a pass SUCCESS "$NOW"),$(chk w fail FAILURE "$NOW")"
expect nofailfast 1 --no-fail-fast
mk cancelled last "$(chk a cancel CANCELLED "$NOW")"
expect cancelled 1
mk pending 0 "$(chk a pending QUEUED "")" 8
mk pending 1 "$(chk a pending IN_PROGRESS "$NOW")" 8
mk pending last "$(chk a pass SUCCESS "$NOW")"
expect pending 0
mk queued last "$(chk a pending QUEUED "")" 8
expect queued 2 --queued-deadline 0
mk running last "$(chk a pending IN_PROGRESS "$OLD")" 8
expect running 2
mk ghfail last "" 1
expect ghfail 3
mkdir -p "$T/empty"
set +e; PATH="$T/empty" "$BASH" "$WAIT" 1 >/dev/null 2>&1; got=$?; set -e
if [[ $got -eq 3 ]]; then echo "ok   no-gh (exit $got)"; else echo "FAIL no-gh: want 3 got $got"; fails=$((fails+1)); fi

[[ $fails -eq 0 ]] && echo "wait-pr-checks: all tests passed" || { echo "wait-pr-checks: $fails failure(s)"; exit 1; }
