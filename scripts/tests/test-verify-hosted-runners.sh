#!/usr/bin/env bash
# Tests for scripts/ci/verify-hosted-runners.sh against small throwaway git repositories.
#
# Passes: hosted images (plain and matrix); each approved, fork-guarded self-hosted route (label array and runs-on
# expression) while private; this repository while private.
# Fails: an unapproved label array or label set, a runs-on expression without an approved set or without a hosted
# fallback, an unexpected literal in the expression, a non-hosted matrix entry, a self-hosted route without the
# same-repository guard, and any approved self-hosted route while the repository is public.
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

GUARD="github.event.pull_request.head.repo.full_name == github.repository"

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

# expect_pass/expect_fail <name> <dir> [visibility, default private]
expect_pass() {
  local name="$1" dir="$2" vis="${3:-private}"
  if REPO_VISIBILITY="${vis}" "${AUDIT}" "${dir}" >/dev/null 2>&1; then ok "${name}"; else ko "${name} (expected pass)"; fi
}
expect_fail() {
  local name="$1" dir="$2" vis="${3:-private}"
  if REPO_VISIBILITY="${vis}" "${AUDIT}" "${dir}" >/dev/null 2>&1; then ko "${name} (expected failure)"; else ok "${name}"; fi
}

make_repo "${WORK}/clean"
expect_pass "clean repository" "${WORK}/clean"
expect_pass "hosted-only repository while public" "${WORK}/clean" public

make_repo "${WORK}/labels"
add_job "${WORK}/labels" "  e:" "    runs-on: [linux, x64]"
expect_fail "unapproved runner label array" "${WORK}/labels"

make_repo "${WORK}/expr"
add_job "${WORK}/expr" "  e:" "    runs-on: >-" "      \${{ vars.X == 1 && fromJSON(1) || 2 }}"
expect_fail "runs-on expression without an approved set" "${WORK}/expr"

make_repo "${WORK}/matrix"
sed -i.bak 's/- os: ubuntu-24.04/- os: my-box/' "${WORK}/matrix/.github/workflows/ci.yml" && rm -f "${WORK}/matrix/.github/workflows/ci.yml.bak"
git -C "${WORK}/matrix" add -A
expect_fail "non-hosted matrix entry" "${WORK}/matrix"

for set in '"self-hosted","macOS","ARM64","local-mac"' '"self-hosted","Windows","X64","local-win"' '"self-hosted","Linux","X64","local-linux"'; do
  label="${set##*,}"; label="${label//\"/}"
  make_repo "${WORK}/expr-${label}"
  add_job "${WORK}/expr-${label}" "  e:" "    runs-on: >-" \
    "      \${{ vars.R == 'on' && inputs.runner != 'hosted' &&" \
    "      (github.event_name != 'pull_request' || ${GUARD})" \
    "      && fromJSON('[${set}]') || 'ubuntu-latest' }}"
  expect_pass "approved ${label} expression while private" "${WORK}/expr-${label}"
  expect_fail "approved ${label} expression while public" "${WORK}/expr-${label}" public
done

make_repo "${WORK}/array-ok"
add_job "${WORK}/array-ok" "  e:" "    if: vars.MAC_RUNNER == 'on' && ${GUARD}" "    runs-on: [self-hosted, macOS, ARM64, local-mac]"
expect_pass "approved, guarded label array while private" "${WORK}/array-ok"

make_repo "${WORK}/no-guard"
add_job "${WORK}/no-guard" "  e:" "    if: vars.MAC_RUNNER == 'on'" "    runs-on: [self-hosted, macOS, ARM64, local-mac]"
expect_fail "self-hosted label array without the same-repository guard" "${WORK}/no-guard"

make_repo "${WORK}/no-guard-expr"
add_job "${WORK}/no-guard-expr" "  e:" "    runs-on: \${{ vars.R == 'on' && fromJSON('[\"self-hosted\",\"Windows\",\"X64\",\"local-win\"]') || 'windows-latest' }}"
expect_fail "self-hosted expression without the same-repository guard" "${WORK}/no-guard-expr"

make_repo "${WORK}/bad-set"
add_job "${WORK}/bad-set" "  e:" "    runs-on: >-" \
  "      \${{ vars.R == 'on' && (${GUARD}) && fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"someone-else\"]') || 'ubuntu-latest' }}"
expect_fail "unapproved label set in an expression" "${WORK}/bad-set"

make_repo "${WORK}/no-fallback"
add_job "${WORK}/no-fallback" "  e:" "    runs-on: >-" \
  "      \${{ (${GUARD}) && fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"local-linux\"]') || vars.OTHER }}"
expect_fail "expression without a hosted fallback" "${WORK}/no-fallback"

make_repo "${WORK}/odd-literal"
add_job "${WORK}/odd-literal" "  e:" "    runs-on: >-" \
  "      \${{ vars.R == 'on' && (${GUARD}) && fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"local-linux\"]') || vars.X == 'y' && 'my-box' || 'ubuntu-latest' }}"
expect_fail "unexpected runner literal in an expression" "${WORK}/odd-literal"

expect_pass "this repository (private)" "${REPO_ROOT}" private

echo "${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
