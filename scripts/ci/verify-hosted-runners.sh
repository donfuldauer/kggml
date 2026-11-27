#!/usr/bin/env bash
# Audits where workflow jobs run: every job must run on a GitHub-hosted runner.
#
# In a public repository a pull request from a fork can edit a workflow, so a self-hosted runner would let anyone run
# code on the machine behind it. Rules:
#   - every `runs-on` is a GitHub-hosted image (ubuntu-*, macos-*, windows-*) or a `${{ matrix.* }}` value;
#   - every matrix runner entry (`os:` / `runner:`) is a GitHub-hosted image.
# Anything else (a label array such as [self-hosted, ...], a runner group, an expression) fails.
#
# Usage: scripts/ci/verify-hosted-runners.sh [<repo-root>]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

hosted_images='^(ubuntu|macos|windows)-[0-9a-z.]+$'
status=0
jobs_checked=0
fail() { echo "FAIL: $*" >&2; status=1; }

# Prints one tab-separated record per job: <job>\t<runs-on value, continuation lines joined>.
jobs_of() {
  awk '
    function flush() { if (job != "") printf "%s\t%s\n", job, ro }
    /^jobs:[[:space:]]*$/ { injobs = 1; next }
    injobs && /^[^[:space:]#]/ { flush(); job = ""; injobs = 0; next }
    injobs && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { flush(); job = $1; sub(/:$/, "", job); ro = ""; cont = 0; next }
    injobs && job != "" {
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

  while IFS=$'\t' read -r job value; do
    [[ -n "${value}" ]] || continue
    jobs_checked=$((jobs_checked + 1))
    value="$(sed -E 's/^["'\'']//; s/["'\'']$//' <<<"${value}")"
    if [[ "${value}" =~ ${hosted_images} ]]; then
      :
    elif [[ "${value}" =~ ^\$\{\{[[:space:]]*matrix\.[a-z_-]+[[:space:]]*\}\}$ ]]; then
      : # matrix entries are checked below
    else
      fail "${wf}: job '${job}': runs-on '${value}' is not a GitHub-hosted image"
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

if [[ ${status} -eq 0 ]]; then
  echo "Runner audit OK: all ${jobs_checked} job(s) run on GitHub-hosted runners."
fi
exit "${status}"
