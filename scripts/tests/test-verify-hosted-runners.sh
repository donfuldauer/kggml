#!/usr/bin/env bash
# Tests for scripts/ci/verify-hosted-runners.sh against small throwaway git repositories.
#
# Asserts the audit passes for workflows that only use GitHub-hosted images (plain and matrix) and for this
# repository, and FAILS for: a runner label array, a runs-on expression, a non-hosted matrix entry, and the former
# private-runner label / variable / keyword mentioned anywhere in a tracked file.
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

expect_pass() {
  local name="$1" dir="$2"
  if "${AUDIT}" "${dir}" >/dev/null 2>&1; then ok "${name}"; else ko "${name} (expected pass)"; fi
}
expect_fail() {
  local name="$1" dir="$2"
  if "${AUDIT}" "${dir}" >/dev/null 2>&1; then ko "${name} (expected failure)"; else ok "${name}"; fi
}

make_repo "${WORK}/clean"
expect_pass "clean repository" "${WORK}/clean"

# Untracked files are not part of the repository and are ignored.
make_repo "${WORK}/untracked"
echo "self""-hosted" >"${WORK}/untracked/scratch.txt"
expect_pass "untracked file ignored" "${WORK}/untracked"

make_repo "${WORK}/labels"
printf '  e:\n    runs-on: [linux, x64]\n' >>"${WORK}/labels/.github/workflows/ci.yml"
git -C "${WORK}/labels" add -A
expect_fail "runner label array" "${WORK}/labels"

make_repo "${WORK}/expr"
printf '  e:\n    runs-on: >-\n      ${{ vars.X == 1 && fromJSON(1) || 2 }}\n' >>"${WORK}/expr/.github/workflows/ci.yml"
git -C "${WORK}/expr" add -A
expect_fail "runs-on expression" "${WORK}/expr"

make_repo "${WORK}/matrix"
sed -i.bak 's/- os: ubuntu-24.04/- os: my-box/' "${WORK}/matrix/.github/workflows/ci.yml" && rm -f "${WORK}/matrix/.github/workflows/ci.yml.bak"
git -C "${WORK}/matrix" add -A
expect_fail "non-hosted matrix entry" "${WORK}/matrix"

for token in "self""-hosted" "local""-mac" "MAC""_RUNNER"; do
  make_repo "${WORK}/tok-${token}"
  echo "See the ${token} runner." >>"${WORK}/tok-${token}/README.md"
  git -C "${WORK}/tok-${token}" add -A
  expect_fail "mention of ${token}" "${WORK}/tok-${token}"
done

expect_pass "this repository" "${REPO_ROOT}"

echo "${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
