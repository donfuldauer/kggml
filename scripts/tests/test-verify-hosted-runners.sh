#!/usr/bin/env bash
# Tests for scripts/ci/verify-hosted-runners.sh against small throwaway git repositories.
#
# Passes: hosted images (plain, quoted and matrix); this repository.
# Fails: any self-hosted label array, a runs-on expression (even one that falls back to a hosted image), a runner
# group, and a non-hosted matrix entry.
#
# Usage: scripts/tests/test-verify-hosted-runners.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT="${REPO_ROOT}/scripts/ci/verify-hosted-runners.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

passed=0
failed=0
ok() { echo "  ok   $1"; passed=$((passed + 1)); }
ko() { echo "  FAIL $1" >&2; failed=$((failed + 1)); }

# make_repo <dir>: a clean repository the audit accepts.
make_repo() {
  local dir="$1"
  mkdir -p "${dir}/.github/workflows"
  git -C "${dir}" init -q
  cat >"${dir}/.github/workflows/ci.yml" <<'YML'
jobs:
  a:
    runs-on: ubuntu-latest
  b:
    runs-on: "macos-14"
  c:
    strategy:
      matrix:
        include:
          - os: windows-latest
          - os: ubuntu-24.04
    runs-on: ${{ matrix.os }}
  d:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
YML
  echo "# Docs" >"${dir}/README.md"
  git -C "${dir}" add -A
}

# add_job <dir> <yaml lines...>: appends a job to the workflow.
add_job() {
  local dir="$1"; shift
  printf '%s\n' "$@" >>"${dir}/.github/workflows/ci.yml"
  git -C "${dir}" add -A
}

# expect_pass/expect_fail <name> <dir>
expect_pass() {
  if "${AUDIT}" "$2" >/dev/null 2>&1; then ok "$1"; else ko "$1 (expected pass)"; fi
}
expect_fail() {
  if "${AUDIT}" "$2" >/dev/null 2>&1; then ko "$1 (expected failure)"; else ok "$1"; fi
}

make_repo "${WORK}/clean"
expect_pass "clean repository" "${WORK}/clean"

make_repo "${WORK}/labels"
add_job "${WORK}/labels" "  e:" "    runs-on: [self-hosted, macOS, ARM64]"
expect_fail "self-hosted label array" "${WORK}/labels"

make_repo "${WORK}/plain-label"
add_job "${WORK}/plain-label" "  e:" "    runs-on: self-hosted"
expect_fail "plain self-hosted label" "${WORK}/plain-label"

make_repo "${WORK}/expr"
add_job "${WORK}/expr" "  e:" "    runs-on: >-" \
  "      \${{ vars.R == 'on' && fromJSON('[\"self-hosted\",\"Linux\",\"X64\"]') || 'ubuntu-latest' }}"
expect_fail "runs-on expression with a hosted fallback" "${WORK}/expr"

make_repo "${WORK}/group"
add_job "${WORK}/group" "  e:" "    runs-on:" "      group: my-runners"
expect_fail "runner group" "${WORK}/group"

make_repo "${WORK}/matrix"
sed -i.bak 's/- os: ubuntu-24.04/- os: my-box/' "${WORK}/matrix/.github/workflows/ci.yml" && rm -f "${WORK}/matrix/.github/workflows/ci.yml.bak"
git -C "${WORK}/matrix" add -A
expect_fail "non-hosted matrix entry" "${WORK}/matrix"

expect_pass "this repository" "${REPO_ROOT}"

echo "${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
