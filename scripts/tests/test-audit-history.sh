#!/usr/bin/env bash
# Tests for scripts/ci/audit-history.sh against small throwaway git repositories.
#
# Asserts the audit passes for a clean history (including a synthetic /home/someone/ path and a binary recorded in
# NOTICE) and FAILS when any commit on any branch ever contained: an oversized blob, a credential/workstation file
# (also after it was deleted again), an unattributed binary, a private-key block, or a developer-machine path (in a
# file or a commit message).
# Also checks that non-no-reply commit e-mails are only reported by default, fail with --strict-identity, and are
# never printed, and that a shallow clone is refused.
#
# Usage: scripts/tests/test-audit-history.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT="${REPO_ROOT}/scripts/ci/audit-history.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

passed=0
failed=0
ok() { echo "  ok   $1"; passed=$((passed + 1)); }
ko() { echo "  FAIL $1" >&2; failed=$((failed + 1)); }

export GIT_AUTHOR_NAME="Test" GIT_AUTHOR_EMAIL="1+test@users.noreply.github.com"
export GIT_COMMITTER_NAME="Test" GIT_COMMITTER_EMAIL="1+test@users.noreply.github.com"
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1

commit() { git -C "$1" add -A && git -C "$1" commit -q -m "$2"; }

# make_repo <dir>: a clean history the audit accepts.
make_repo() {
  local dir="$1"
  mkdir -p "${dir}/fixtures"
  git -C "${dir}" init -q -b main
  echo '# Demo' >"${dir}/README.md"
  echo 'echo "Built in /home/someone/demo."' >"${dir}/build.sh"
  printf '\x00\x01\x02' >"${dir}/fixtures/tone.wav"
  printf '| Fixture | License |\n| `fixtures/tone.wav` | CC0-1.0 |\n' >"${dir}/NOTICE"
  commit "${dir}" "initial"
}

run_audit() { "${AUDIT}" --repo "$1" --skip-gitleaks "${@:2}"; }
expect_pass() {
  local name="$1" dir="$2"
  if run_audit "${dir}" "${@:3}" >/dev/null 2>&1; then ok "${name}"; else ko "${name} (expected pass)"; fi
}
expect_fail() {
  local name="$1" dir="$2"
  if run_audit "${dir}" "${@:3}" >/dev/null 2>&1; then ko "${name} (expected failure)"; else ok "${name}"; fi
}

make_repo "${WORK}/clean"
expect_pass "clean history" "${WORK}/clean"

# A file added and deleted again is still in the history.
make_repo "${WORK}/deleted"
echo 'sdk.dir=/opt/sdk' >"${WORK}/deleted/local.properties"
commit "${WORK}/deleted" "oops"
git -C "${WORK}/deleted" rm -q local.properties
commit "${WORK}/deleted" "remove"
expect_fail "deleted local.properties still in history" "${WORK}/deleted"

# Findings on a side branch count too (every ref is scanned).
make_repo "${WORK}/branch"
git -C "${WORK}/branch" checkout -q -b side
printf 'k' >"${WORK}/branch/release.jks"
commit "${WORK}/branch" "keystore"
git -C "${WORK}/branch" checkout -q main
expect_fail "keystore on another branch" "${WORK}/branch"

make_repo "${WORK}/big"
head -c 2048 /dev/zero >"${WORK}/big/blob.txt"
commit "${WORK}/big" "big"
if KGGML_HISTORY_MAX_BLOB_BYTES=1024 run_audit "${WORK}/big" >/dev/null 2>&1; then
  ko "oversized blob (expected failure)"
else
  ok "oversized blob"
fi
expect_pass "blob under the default size limit" "${WORK}/big"

make_repo "${WORK}/model"
mkdir -p "${WORK}/model/models"
printf 'GGUF' >"${WORK}/model/models/tiny.gguf"
commit "${WORK}/model" "model"
git -C "${WORK}/model" rm -q models/tiny.gguf
commit "${WORK}/model" "drop model"
expect_fail "unattributed model file in history" "${WORK}/model"

make_repo "${WORK}/key"
printf -- '-----BEGIN OPENSSH PRIVATE ''KEY-----\nabc\n-----END OPENSSH PRIVATE ''KEY-----\n' >"${WORK}/key/notes.txt"
commit "${WORK}/key" "key"
expect_fail "private key block" "${WORK}/key"

make_repo "${WORK}/macpath"
# Paths are built from pieces so this test file does not itself match the audit.
echo 'path = "/Users/''alice/dev/demo"' >>"${WORK}/macpath/build.sh"
commit "${WORK}/macpath" "path"
expect_fail "macOS home path" "${WORK}/macpath"

make_repo "${WORK}/linuxpath"
echo 'cd /home/''bob/work' >>"${WORK}/linuxpath/build.sh"
commit "${WORK}/linuxpath" "path"
expect_fail "Linux home path" "${WORK}/linuxpath"

make_repo "${WORK}/winpath"
echo 'set P=C:\Users\''carol\src' >>"${WORK}/winpath/build.sh"
commit "${WORK}/winpath" "path"
expect_fail "Windows home path" "${WORK}/winpath"

make_repo "${WORK}/rootpath"
echo 'cp out /root''/build' >>"${WORK}/rootpath/build.sh"
commit "${WORK}/rootpath" "path"
expect_fail "root home path" "${WORK}/rootpath"

# Commit messages are scanned too (gitleaks only reads diffs).
make_repo "${WORK}/msgpath"
git -C "${WORK}/msgpath" commit -q --allow-empty -m "fix build on /Users/""dave/src/kggml"
expect_fail "developer path in a commit message" "${WORK}/msgpath"

make_repo "${WORK}/msgkey"
git -C "${WORK}/msgkey" commit -q --allow-empty -m "$(printf -- '-----BEGIN RSA PRIVATE ''KEY-----\nabc')"
expect_fail "private key block in a commit message" "${WORK}/msgkey"

make_repo "${WORK}/msgwin"
git -C "${WORK}/msgwin" commit -q --allow-empty -m 'build in C:\Users\''erin\src'
expect_fail "Windows path in a commit message" "${WORK}/msgwin"

make_repo "${WORK}/msgok"
git -C "${WORK}/msgok" commit -q --allow-empty -m "docs: e.g. /home/""someone/kggml"
expect_pass "placeholder path in a commit message" "${WORK}/msgok"

# A personal commit e-mail is reported (never printed) and only fails with --strict-identity.
make_repo "${WORK}/identity"
echo more >>"${WORK}/identity/README.md"
GIT_AUTHOR_EMAIL="dev@private-mail.test" commit "${WORK}/identity" "personal address"
expect_pass "personal e-mail is only reported" "${WORK}/identity"
expect_fail "personal e-mail fails with --strict-identity" "${WORK}/identity" --strict-identity
if run_audit "${WORK}/identity" 2>&1 | grep -qF "private-mail.test"; then
  ko "e-mail address printed"
else
  ok "e-mail address not printed"
fi

# A shallow clone cannot be audited.
git -C "${WORK}/identity" commit -q --allow-empty -m "second"
git clone -q --depth 1 "file://${WORK}/identity" "${WORK}/shallow"
if run_audit "${WORK}/shallow" >/dev/null 2>&1; then ko "shallow clone (expected refusal)"; else ok "shallow clone refused"; fi

echo "audit-history tests: ${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
