#!/usr/bin/env bash
# Audits where workflow jobs run (issues #23, #62, #63).
#
# While the repository is private, the owner's own machines may serve as self-hosted runners, but only these label
# sets (AGENTS.md "Runner order"):
#   [self-hosted, macOS, ARM64, local-mac]      repo variable MAC_RUNNER
#   [self-hosted, Windows, X64, local-win]      repo variable WIN_RUNNER
#   [self-hosted, Linux, X64, local-linux]      repo variable LINUX_RUNNER
# Rules:
#   - every `runs-on` is a GitHub-hosted image (ubuntu-*, macos-*, windows-*), a `${{ matrix.* }}` value whose matrix
#     entries are hosted images, one of the label sets above, or an expression whose only runner outcomes are
#     fromJSON('<one of the label sets>') and a final `|| '<hosted image>'` fallback;
#   - every job that can reach a self-hosted runner carries the same-repository guard
#     `github.event.pull_request.head.repo.full_name == github.repository`, so a fork PR never reaches one;
#   - when the repository is PUBLIC, any self-hosted route fails the audit: a fork PR could edit a workflow to run
#     arbitrary code on a private machine, so every runner must be removed first (issue #63).
# The visibility comes from $REPO_VISIBILITY (CI passes github.event.repository.visibility), else `gh repo view`, else
# it is reported as unknown and only the structural rules apply.
#
# Usage: scripts/ci/verify-hosted-runners.sh [<repo-root>]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

hosted_images='^(ubuntu|macos|windows)-[0-9a-z.]+$'
allowed_sets=(
  "self-hosted,macOS,ARM64,local-mac"
  "self-hosted,Windows,X64,local-win"
  "self-hosted,Linux,X64,local-linux"
)
fork_guard='github.event.pull_request.head.repo.full_name == github.repository'

visibility="${REPO_VISIBILITY:-}"
if [[ -z "${visibility}" ]] && command -v gh >/dev/null 2>&1; then
  visibility="$(gh repo view --json visibility --jq .visibility 2>/dev/null || true)"
fi
visibility="$(tr '[:upper:]' '[:lower:]' <<<"${visibility:-unknown}")"

status=0
private_uses=()

# "[ 'a', \"b\" ,c ]" -> "a,b,c"
normalize_set() { tr -d " '\"[]" <<<"$1"; }
is_allowed_set() {
  local s
  s="$(normalize_set "$1")"
  for a in "${allowed_sets[@]}"; do [[ "${s}" == "${a}" ]] && return 0; done
  return 1
}
fail() { echo "FAIL: $*" >&2; status=1; }

# Prints one tab-separated record per job: <job>\t<runs-on value, continuation lines joined>\t<1 if the job block
# contains the fork guard, else 0>.
jobs_of() {
  awk -v guard="${fork_guard}" '
    function flush() { if (job != "") printf "%s\t%s\t%d\n", job, ro, g }
    /^jobs:[[:space:]]*$/ { injobs = 1; next }
    injobs && /^[^[:space:]#]/ { flush(); job = ""; injobs = 0; next }
    injobs && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ {
      flush(); job = $1; sub(/:$/, "", job); ro = ""; g = 0; cont = 0; next
    }
    injobs && job != "" {
      if (index($0, guard)) g = 1
      if (cont) {
        if ($0 ~ /^      +[^ ]/) { l = $0; gsub(/^ +/, "", l); ro = (ro == "" ? l : ro " " l); next }
        cont = 0
      }
      if ($0 ~ /^    runs-on:/) {
        v = $0; sub(/^    runs-on:[[:space:]]*/, "", v); sub(/[[:space:]]+#.*$/, "", v)
        if (v == ">-" || v == ">" || v == "|" || v == "") { ro = ""; cont = 1 } else ro = v
      }
    }
    END { flush() }
  ' "$1"
}

for wf in .github/workflows/*.yml .github/workflows/*.yaml; do
  [[ -f "${wf}" ]] || continue

  while IFS=$'\t' read -r job value has_guard; do
    [[ -n "${value}" ]] || continue
    value="$(sed -E 's/^["'\'']//; s/["'\'']$//' <<<"${value}")"
    self_hosted=0
    if [[ "${value}" =~ ${hosted_images} ]]; then
      :
    elif [[ "${value}" =~ ^\$\{\{[[:space:]]*matrix\.[a-z_-]+[[:space:]]*\}\}$ ]]; then
      : # matrix entries are checked below
    elif [[ "${value}" == \[*\] ]]; then
      if is_allowed_set "${value}"; then self_hosted=1; else fail "${wf}: job '${job}': runner labels ${value} are not an approved self-hosted set"; fi
    elif [[ "${value}" =~ ^\$\{\{.*\}\}$ ]]; then
      if ! [[ "${value}" =~ \|\|[[:space:]]*\'(ubuntu|macos|windows)-[0-9a-z.]+\'[[:space:]]*\}\}$ ]]; then
        fail "${wf}: job '${job}': runs-on expression must end with \"|| '<GitHub-hosted image>'\": ${value}"
      fi
      sets="$(grep -oE "fromJSON\('\[[^]]*\]'\)" <<<"${value}" || true)"
      if [[ -z "${sets}" ]]; then
        fail "${wf}: job '${job}': runs-on expression has no approved fromJSON label set: ${value}"
      fi
      while IFS= read -r s; do
        [[ -n "${s}" ]] || continue
        s="${s#fromJSON(\'}"; s="${s%\')}"
        if is_allowed_set "${s}"; then self_hosted=1; else fail "${wf}: job '${job}': runner labels ${s} are not an approved self-hosted set"; fi
      done <<<"${sets}"
      # Every other quoted literal must be a hosted image or a known non-runner operand.
      rest="$(sed -E "s/fromJSON\('\[[^]]*\]'\)//g" <<<"${value}")"
      while IFS= read -r lit; do
        lit="${lit//\'/}"
        [[ -z "${lit}" || "${lit}" =~ ${hosted_images} || "${lit}" =~ ^(on|hosted|pull_request)$ ]] && continue
        fail "${wf}: job '${job}': unexpected literal '${lit}' in runs-on expression"
      done < <(grep -oE "'[^']*'" <<<"${rest}" || true)
    else
      fail "${wf}: job '${job}': runs-on '${value}' is not a GitHub-hosted image or an approved self-hosted route"
    fi
    if [[ ${self_hosted} -eq 1 ]]; then
      private_uses+=("${wf}: ${job}")
      if [[ "${has_guard}" != "1" ]]; then
        fail "${wf}: job '${job}' can run on a self-hosted runner but lacks the same-repository guard (${fork_guard})"
      fi
    fi
  done < <(jobs_of "${wf}")

  # Matrix runner entries (e.g. `- os: windows-latest` / `os: [ubuntu-latest, ...]`) must be hosted images.
  while IFS= read -r line; do
    for value in $(sed -E 's/^[^:]*:[[:space:]]*-?[[:space:]]*(os|runner):[[:space:]]*//; s/[][,"'\'']/ /g' <<<"${line}"); do
      if ! [[ "${value}" =~ ${hosted_images} ]]; then
        fail "${wf}: matrix runner '${value}' is not a GitHub-hosted image"
      fi
    done
  done < <(grep -nE '^[[:space:]]*-?[[:space:]]*(os|runner):[[:space:]]*' "${wf}" || true)
done

if [[ ${#private_uses[@]} -gt 0 ]]; then
  if [[ "${visibility}" == "public" ]]; then
    fail "the repository is public but these jobs can still run on self-hosted runners; remove every runner first (issue #63):"
    printf '  %s\n' "${private_uses[@]}" >&2
  elif [[ "${visibility}" != "private" && "${visibility}" != "internal" ]]; then
    echo "WARN: repository visibility unknown (set REPO_VISIBILITY or sign in to gh); self-hosted routes not checked against it." >&2
  fi
fi

if [[ ${status} -eq 0 ]]; then
  echo "Runner audit OK (${visibility}): ${#private_uses[@]} job(s) with an approved, fork-guarded self-hosted route; every other job runs on a GitHub-hosted image."
fi
exit "${status}"
