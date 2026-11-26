#!/usr/bin/env bash
# Fails if any tracked file (outside vendored submodules) reintroduces a private CI runner: a `runs-on` that is not
# a GitHub-hosted image, the owner's former Mac runner label, or its on/off repository variable. Before the
# repository is public every job must run on GitHub-hosted runners, because a fork PR can edit a workflow to run
# arbitrary code on any runner registered to the repository (issue #23).
#
# Usage: scripts/ci/verify-hosted-runners.sh [<repo-root>]
set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

# Built from pieces so this script does not match itself.
forbidden=("self""-hosted" "local""-mac" "MAC""_RUNNER")
hosted_images='^(ubuntu|macos|windows)-[0-9a-z.]+$'

status=0
files=()
while IFS= read -r -d '' f; do files+=("$f"); done < <(git ls-files -z)

for token in "${forbidden[@]}"; do
  hits="$(printf '%s\0' "${files[@]}" | xargs -0 grep -nF -- "${token}" 2>/dev/null || true)"
  if [[ -n "${hits}" ]]; then
    echo "FAIL: '${token}' found (private runners are not allowed):" >&2
    echo "${hits}" >&2
    status=1
  fi
done

# Every runs-on in our own workflows must be a single GitHub-hosted image name (no labels arrays, no expressions).
for wf in .github/workflows/*.yml .github/workflows/*.yaml; do
  [[ -f "${wf}" ]] || continue
  while IFS= read -r line; do
    num="${line%%:*}"
    value="$(sed -E 's/^[0-9]+:[[:space:]]*runs-on:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^["'\'']//; s/["'\'']$//' <<<"${line}")"
    if [[ "${value}" =~ ^\$\{\{[[:space:]]*matrix\.[a-z_-]+[[:space:]]*\}\}$ ]]; then
      # A matrix value: every `os:`-style entry of the workflow is checked below.
      continue
    fi
    if ! [[ "${value}" =~ ${hosted_images} ]]; then
      echo "FAIL: ${wf}:${num}: runs-on '${value}' is not a GitHub-hosted image (ubuntu-*, macos-*, windows-*)" >&2
      status=1
    fi
  done < <(grep -nE '^[[:space:]]*runs-on:' "${wf}" || true)
  # Matrix runner entries (e.g. `- os: windows-latest` / `os: [ubuntu-latest, ...]`).
  while IFS= read -r line; do
    for value in $(sed -E 's/^[^:]*:[[:space:]]*-?[[:space:]]*(os|runner):[[:space:]]*//; s/[][,"'\'']/ /g' <<<"${line}"); do
      if ! [[ "${value}" =~ ${hosted_images} ]]; then
        echo "FAIL: ${wf}: matrix runner '${value}' is not a GitHub-hosted image" >&2
        status=1
      fi
    done
  done < <(grep -nE '^[[:space:]]*-?[[:space:]]*(os|runner):[[:space:]]*' "${wf}" || true)
done

if [[ ${status} -eq 0 ]]; then
  echo "Hosted-runner audit OK: every workflow job runs on a GitHub-hosted image; no private runner references."
fi
exit "${status}"
