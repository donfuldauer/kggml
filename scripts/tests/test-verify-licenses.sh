#!/usr/bin/env bash
# Tests for scripts/ci/verify-licenses.sh against small throwaway git repositories.
#
# Asserts the audit passes for a clean repository (and for this repository), and FAILS for: a modified MIT text,
# a malformed copyright line, a submodule pin / path / URL missing from NOTICE, an unrecorded FetchContent pin,
# a stale commit SHA in NOTICE, an unrecorded binary or model file, a non-MIT/CC0 fixture license, a missing
# committed fixture and a missing fixture generator source.
#
# Usage: scripts/tests/test-verify-licenses.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT="${REPO_ROOT}/scripts/ci/verify-licenses.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

passed=0
failed=0
ok() { echo "  ok   $1"; passed=$((passed + 1)); }
ko() { echo "  FAIL $1" >&2; failed=$((failed + 1)); }

SUB_SHA=1111111111111111111111111111111111111111
GT_SHA=2222222222222222222222222222222222222222
STALE_SHA=3333333333333333333333333333333333333333

# make_repo <dir>: a clean repository the audit accepts.
make_repo() {
  local dir="$1"
  mkdir -p "${dir}"
  git -C "${dir}" init -q
  git -C "${dir}" config user.email test@example.invalid
  git -C "${dir}" config user.name test
  git -C "${dir}" config core.hooksPath /dev/null
  cp "${REPO_ROOT}/LICENSE" "${dir}/LICENSE"
  sed -i.bak '3s/.*/Copyright (c) 2026 Example Holder/' "${dir}/LICENSE" && rm -f "${dir}/LICENSE.bak"
  cat >"${dir}/.gitmodules" <<'EOF'
[submodule "lib/third_party/up"]
	path = lib/third_party/up
	url = https://github.com/example/up.git
EOF
  mkdir -p "${dir}/lib/tests" "${dir}/lib/fixtures"
  cat >"${dir}/lib/tests/CMakeLists.txt" <<EOF
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        ${GT_SHA} # v1.15.2
)
EOF
  echo "// generator" >"${dir}/lib/tests/gen.cpp"
  printf 'RIFF\0\0\0\0WAVE' >"${dir}/lib/fixtures/clip.wav"
  cat >"${dir}/NOTICE" <<EOF
# Third-Party Notices

| Component | Upstream Repository | License | Pinned Commit / Tag | Module |
| --- | --- | --- | --- | --- |
| up | https://github.com/example/up | MIT | \`${SUB_SHA}\` (submodule \`lib/third_party/up\`) | lib |
| GoogleTest | https://github.com/google/googletest | BSD-3-Clause | \`${GT_SHA}\` | tests |

## Test Fixtures & Micro-Weights

| Fixture Path | Source / Origin | License | Notes |
| --- | --- | --- | --- |
| \`lib/fixtures/clip.wav\` | Recorded by the author | CC0-1.0 | committed clip |
| *(generated at test time)* model.bin | Written by \`lib/tests/gen.cpp\` | MIT (Kggml) | generated |
EOF
  git -C "${dir}" add -A
  git -C "${dir}" update-index --add --cacheinfo "160000,${SUB_SHA},lib/third_party/up"
  git -C "${dir}" commit -q -m init
}

# expect <pass|fail> <name> <expected-stderr-regex or ''> <mutation command run in the repo dir or ''>
expect() {
  local want="$1" name="$2" regex="$3" mutate="$4" dir out rc
  dir="${WORK}/${name}"
  make_repo "${dir}"
  if [[ -n "${mutate}" ]]; then
    (cd "${dir}" && eval "${mutate}" && git add -A -- . ':!lib/third_party/up' && git commit -q -m mutate --allow-empty)
  fi
  set +e
  out="$(KGGML_LICENSE_AUDIT_ROOT="${dir}" "${AUDIT}" 2>&1)"
  rc=$?
  set -e
  if [[ "${want}" == pass && ${rc} -eq 0 ]]; then ok "${name}"; return; fi
  if [[ "${want}" == fail && ${rc} -ne 0 ]] && { [[ -z "${regex}" ]] || grep -qE -- "${regex}" <<<"${out}"; }; then
    ok "${name}"; return
  fi
  ko "${name} (expected ${want}, rc=${rc})"; echo "${out}" | sed 's/^/       /' >&2
}

echo "verify-licenses.sh:"
expect pass clean '' ''
expect fail license-text 'not the standard MIT' "sed -i.bak 's/free of charge/for a fee/' LICENSE && rm LICENSE.bak"
expect fail copyright-line "line 3 must be" "sed -i.bak '3s/.*/Copyright Example/' LICENSE && rm LICENSE.bak"
expect fail submodule-sha 'pinned at 4444' \
  "git update-index --cacheinfo 160000,4444444444444444444444444444444444444444,lib/third_party/up"
expect fail submodule-path 'submodule path lib/third_party/up' "sed -i.bak 's|(submodule .*) | |' NOTICE && rm NOTICE.bak"
expect fail submodule-url 'upstream https://github.com/example/up is not' \
  "sed -i.bak 's|https://github.com/example/up |https://example.invalid/other |' NOTICE && rm NOTICE.bak"
expect fail fetchcontent 'pins GIT_TAG 5555' \
  "sed -i.bak 's/${GT_SHA}/5555555555555555555555555555555555555555/' lib/tests/CMakeLists.txt && rm lib/tests/CMakeLists.txt.bak"
expect fail stale-sha "records commit ${STALE_SHA}" "echo 'old pin ${STALE_SHA}' >>NOTICE"
expect fail unrecorded-binary 'blob.dat is not recorded' "printf '\\0\\1\\2\\0' >lib/blob.dat"
expect fail unrecorded-model 'tiny.gguf is not recorded' "echo text >lib/tiny.gguf"
expect fail fixture-license 'not MIT or CC0 \(GPL-3.0\)' "sed -i.bak 's/| CC0-1.0 |/| GPL-3.0 |/' NOTICE && rm NOTICE.bak"
expect fail fixture-missing 'clip.wav in NOTICE is not tracked' "git rm -q lib/fixtures/clip.wav && printf '\\0' >lib/other.bin && echo '\`lib/other.bin\`' >>NOTICE"
expect fail generator-missing 'lib/tests/gen.cpp in NOTICE does not exist' "git rm -q lib/tests/gen.cpp"

echo "this repository:"
if "${AUDIT}" >/dev/null 2>&1; then ok repository; else ko repository; "${AUDIT}" || true; fi

echo "${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
