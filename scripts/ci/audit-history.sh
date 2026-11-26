#!/usr/bin/env bash
# Full-history pre-public audit (issue #24). Scans every commit reachable from any ref (all branches, tags and, if
# fetched, pull-request heads), not just the current tree, and fails if any of them ever contained:
#   1. a secret gitleaks detects (full-history `gitleaks git --log-opts=--all`, using .gitleaks.toml);
#   2. a blob larger than the size limit (default 1 MiB; KGGML_HISTORY_MAX_BLOB_BYTES);
#   3. a key, keystore, credential or workstation file (.pem, .key, .p12, .pfx, .jks, .keystore, .gpg, .env other
#      than .env.example,
#      local.properties, .DS_Store, id_rsa/id_ed25519, ...), at any path in any commit;
#   4. a model / audio / image / archive / binary-data file whose path is not recorded in the current NOTICE (the
#      same extension list as scripts/ci/verify-licenses.sh, applied to history instead of the index);
#   5. a private-key block (`-----BEGIN ... PRIVATE KEY-----`, PGP private key block) in any blob or commit message;
#   6. a developer-machine path (/Users/<name>/, /home/<name>/, the root home directory, C:\Users\<name>\) in any blob or commit
#      message, except the synthetic
#      names in KGGML_HISTORY_PATH_ALLOW (default: someone, example, runner, username).
# It also reports, without failing, commits whose author or committer e-mail is not a GitHub/Anthropic no-reply
# address (the address itself is never printed). With --strict-identity those fail too.
#
# Remediation for a finding is a history rewrite with `git filter-repo`, which only the repository owner may run:
# see docs/OSS_LAUNCH.md ("Remediating a finding").
#
# Needs a full clone (`actions/checkout` with `fetch-depth: 0`). Shared by .github/workflows/repo-hygiene.yml and
# ./scripts/ci-local.sh (step `history`).
#
# Usage: scripts/ci/audit-history.sh [--repo <dir>] [--skip-gitleaks] [--strict-identity]
#   --skip-gitleaks    do not run gitleaks (used by the tests; CI and ci-local.sh always run it)
#   --strict-identity  fail on a non-no-reply commit e-mail instead of reporting it
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
skip_gitleaks=0
strict_identity=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) ROOT="$(cd "${2:?--repo needs a directory}" && pwd)"; shift ;;
    --skip-gitleaks) skip_gitleaks=1 ;;
    --strict-identity) strict_identity=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
cd "${ROOT}"

max_blob="${KGGML_HISTORY_MAX_BLOB_BYTES:-1048576}"
path_allow="${KGGML_HISTORY_PATH_ALLOW:-someone|example|runner|username}"

errors=0
err() { echo "history audit: FAIL: $*" >&2; errors=$((errors + 1)); }
note() { echo "history audit: $*"; }

if [[ "$(git rev-parse --is-shallow-repository)" == "true" ]]; then
  echo "history audit: the clone is shallow; fetch the full history first (actions/checkout fetch-depth: 0)" >&2
  exit 2
fi

revs_file="$(mktemp)"
objs_file="$(mktemp)"
trap 'rm -f "${revs_file}" "${objs_file}"' EXIT

git rev-list --all >"${revs_file}"
commit_count="$(wc -l <"${revs_file}" | tr -d ' ')"
if [[ "${commit_count}" -eq 0 ]]; then
  echo "history audit: no commits found" >&2
  exit 2
fi

# Every blob ever reachable, with the first path it was seen at: "<sha> <size> <path>".
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1 == "blob" { $1 = ""; sub(/^ /, ""); print }' >"${objs_file}"
blob_count="$(wc -l <"${objs_file}" | tr -d ' ')"

# Every path ever committed (rename-aware listing would miss nothing more: --objects already lists each tree entry).
all_paths="$(git log --all --format= --name-only --diff-filter=AMR --no-renames | sed '/^$/d' | sort -u)"

# 1. gitleaks over the whole history.
if [[ ${skip_gitleaks} -eq 0 ]]; then
  if ! command -v gitleaks >/dev/null 2>&1; then
    echo "history audit: gitleaks is not installed (https://github.com/gitleaks/gitleaks)" >&2
    exit 2
  fi
  config=()
  [[ -f .gitleaks.toml ]] && config=(--config .gitleaks.toml)
  if gitleaks git --redact --no-banner --log-level warn "${config[@]}" --log-opts="--all" .; then
    note "gitleaks: no leaks in ${commit_count} commits"
  else
    err "gitleaks found secrets in the history (redacted findings above)"
  fi
fi

# 2. Oversized blobs.
while read -r sha size path; do
  if [[ "${size}" -gt "${max_blob}" ]]; then
    err "blob ${sha:0:12} (${size} bytes > ${max_blob}) at '${path}' is in the history"
  fi
done <"${objs_file}"

# 3. Keys, keystores, credentials and workstation files, at any path ever committed.
forbidden_re='(^|/)(local\.properties|\.DS_Store|\.env(\..*)?|id_(rsa|dsa|ecdsa|ed25519)|.*\.(pem|key|p12|pfx|jks|keystore|gpg|kdbx|mobileprovision|p8)|secring\..*|credentials(\.json)?|google-services\.json|GoogleService-Info\.plist)$'
while IFS= read -r path; do
  [[ -n "${path}" ]] || continue
  err "credential / workstation file '${path}' is in the history"
done < <(grep -E "${forbidden_re}" <<<"${all_paths}" | grep -vE '(^|/)\.env\.example$' || true)

# 4. Binary-data files must be attributed in NOTICE (by path), wherever they appear in history.
data_ext='\.(bin|gguf|ggml|safetensors|ckpt|pt|pth|onnx|tflite|mlmodel|mlpackage|npy|npz|wav|pcm|raw|f32|mp3|flac|ogg|m4a|png|jpe?g|gif|webp|bmp|ico|icns|zip|jar|aar|tar|gz|tgz|xz|7z|so|dylib|dll|a|o|exe|class|metallib|apk|aab|ipa)$'
while IFS= read -r path; do
  [[ -n "${path}" ]] || continue
  if [[ ! -f NOTICE ]] || ! grep -qF -- "\`${path}\`" NOTICE; then
    err "binary/data file '${path}' is in the history but not recorded in NOTICE"
  fi
done < <(grep -iE "${data_ext}" <<<"${all_paths}" || true)

# 5 + 6. Content patterns, searched in every commit's tree (git grep reads blobs directly; no checkout needed) and in
# every commit message. Patterns are built from pieces so this script does not match itself.
key_re='-----BEGIN ([A-Z]+ )?PRIVATE ''KEY-----|-----BEGIN PGP PRIVATE ''KEY BLOCK-----'
home_re='(/Users/|/home/)[A-Za-z0-9._-]+/|/ro''ot/|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+'

# grep_trees <git-grep-args...>: runs git grep over every commit, 64 commits per call, printing
# "<path>:<rest>" once per distinct line (the commit prefix is dropped). git grep exits 1 for "no match";
# anything else is an error and fails the audit instead of passing silently.
grep_trees() {
  local -a batch=()
  local rev rc
  {
    while IFS= read -r rev; do
      batch+=("${rev}")
      if [[ ${#batch[@]} -eq 64 ]]; then
        rc=0; git grep -I "$@" "${batch[@]}" || rc=$?
        [[ ${rc} -le 1 ]] || { echo "history audit: git grep failed (exit ${rc})" >&2; return 2; }
        batch=()
      fi
    done <"${revs_file}"
    if [[ ${#batch[@]} -gt 0 ]]; then
      rc=0; git grep -I "$@" "${batch[@]}" || rc=$?
      [[ ${rc} -le 1 ]] || { echo "history audit: git grep failed (exit ${rc})" >&2; return 2; }
    fi
  } | sed -E 's/^[0-9a-f]{40}://' | sort -u
  return "${PIPESTATUS[0]}"
}

# dev_path_ok <match>: true if the path's user name is one of the synthetic placeholders.
dev_path_ok() {
  local name
  [[ "$1" == "/ro""ot/" ]] && return 1
  name="$(sed -E 's#^(/Users/|/home/)##; s#^[A-Za-z]:\\Users\\##; s#[/\\]+$##' <<<"$1")"
  [[ "${name}" =~ ^(${path_allow})$ ]]
}

key_hits="$(grep_trees -l -E -e "${key_re}")" || { echo "history audit: content scan failed" >&2; exit 2; }
while IFS= read -r hit; do
  [[ -n "${hit}" ]] || continue
  err "private key block in history: ${hit}"
done <<<"${key_hits}"

path_hits="$(grep_trees -o -E -e "${home_re}")" || { echo "history audit: content scan failed" >&2; exit 2; }
while IFS= read -r hit; do
  [[ -n "${hit}" ]] || continue
  match="${hit#*:}"
  dev_path_ok "${match}" && continue
  err "developer-machine path '${match}' in history: ${hit%%:*}"
done <<<"${path_hits}"

# Commit messages (gitleaks only scans diffs).
while IFS=$'\t' read -r h match; do
  [[ -n "${h}" ]] || continue
  if [[ "${match}" == KEY ]]; then
    err "private key block in the message of commit ${h}"
  elif ! dev_path_ok "${match}"; then
    err "developer-machine path '${match}' in the message of commit ${h}"
  fi
done < <(git log --all --format='@@kggml-commit %h%n%B' | KEY_RE="${key_re}" HOME_RE="${home_re}" awk '
  BEGIN { key = ENVIRON["KEY_RE"]; home = ENVIRON["HOME_RE"] }
  /^@@kggml-commit [0-9a-f]+$/ { h = $2; next }
  {
    if ($0 ~ key) print h "\tKEY"
    msg = $0
    while (match(msg, home)) { print h "\t" substr(msg, RSTART, RLENGTH); msg = substr(msg, RSTART + RLENGTH) }
  }' | sort -u)

# Commit identities: report (never print) non-no-reply addresses.
noreply_re='(@users\.noreply\.github\.com|^noreply@github\.com|^noreply@anthropic\.com|\[bot\]@users\.noreply\.github\.com)$'
identity_commits="$(git log --all --format='%h %ae %ce' | while read -r h ae ce; do
  if ! [[ "${ae}" =~ ${noreply_re} && "${ce}" =~ ${noreply_re} ]]; then echo "${h}"; fi
done)"
if [[ -n "${identity_commits}" ]]; then
  n="$(wc -l <<<"${identity_commits}" | tr -d ' ')"
  msg="${n} commit(s) carry an author/committer e-mail that is not a no-reply address (address not shown): $(head -n 20 <<<"${identity_commits}" | tr '\n' ' ')"
  if [[ ${strict_identity} -eq 1 ]]; then
    err "${msg}"
  else
    note "NOTE: ${msg}— owner decision, see docs/OSS_LAUNCH.md (\"Commit e-mail addresses\")"
  fi
fi

if [[ ${errors} -gt 0 ]]; then
  echo "history audit: ${errors} finding(s) in ${commit_count} commits / ${blob_count} blobs" >&2
  exit 1
fi
note "OK: ${commit_count} commits, ${blob_count} blobs, no secrets, oversized blobs, credential files, unattributed binaries, private keys or developer-machine paths"
