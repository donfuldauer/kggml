#!/usr/bin/env bash
# License, upstream-pin and test-fixture audit (issue #21). Fails if:
#   1. LICENSE is not the standard MIT License text (only the copyright line may vary, and it must name a year
#      and a holder);
#   2. a git submodule's pinned commit, path or upstream URL (.gitmodules) is not recorded in NOTICE;
#   3. a FetchContent GIT_TAG commit SHA in a tracked CMake file is not recorded in NOTICE;
#   4. NOTICE records a 40-hex commit SHA that is no longer pinned anywhere (stale upstream hash);
#   5. a tracked binary file, or a file with a model / audio / image / archive extension, is not recorded by path
#      in NOTICE (no unattributed binaries);
#   6. a row of NOTICE's "Test Fixtures & Micro-Weights" table carries a license that is not MIT or CC0, names a
#      committed fixture path that does not exist, or points at a generator source file that does not exist.
# Reads only the git index (submodules need not be checked out). Shared by .github/workflows/repo-hygiene.yml and
# ./scripts/ci-local.sh. KGGML_LICENSE_AUDIT_ROOT overrides the repository root (used by its tests).
#
# Usage: scripts/ci/verify-licenses.sh
set -euo pipefail

ROOT="${KGGML_LICENSE_AUDIT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

errors=0
err() { echo "license audit: $*" >&2; errors=$((errors + 1)); }
notice_has() { grep -qF -- "$1" NOTICE; }

[[ -f NOTICE ]] || { echo "license audit: NOTICE is missing" >&2; exit 1; }

# 1. LICENSE is the exact MIT text.
expected_mit_body() {
  cat <<'EOF'
MIT License

COPYRIGHT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
}
if [[ ! -f LICENSE ]]; then
  err "LICENSE is missing"
else
  copyright="$(sed -n '3p' LICENSE)"
  if ! [[ "${copyright}" =~ ^Copyright\ \(c\)\ [0-9]{4}(-[0-9]{4})?\ [^[:space:]].*$ ]]; then
    err "LICENSE line 3 must be 'Copyright (c) <year> <holder>', got: ${copyright}"
  fi
  if ! diff -u <(expected_mit_body) <(sed '3s/.*/COPYRIGHT/' LICENSE) >&2; then
    err "LICENSE is not the standard MIT License text (diff above)"
  fi
fi

# 2. Submodules: pinned SHA, path and URL recorded in NOTICE.
allowed_shas=()
while read -r mode sha _stage path; do
  [[ "${mode}" == 160000 ]] || continue
  allowed_shas+=("${sha}")
  notice_has "${sha}" || err "submodule ${path} is pinned at ${sha}, which NOTICE does not record"
  notice_has "${path}" || err "submodule path ${path} is not recorded in NOTICE"
  url="$(git config -f .gitmodules --get "submodule.${path}.url" 2>/dev/null || true)"
  url="${url%.git}"
  if [[ -z "${url}" ]]; then
    err "submodule ${path} has no url in .gitmodules"
  elif ! notice_has "${url}"; then
    err "submodule ${path} upstream ${url} is not recorded in NOTICE"
  fi
done < <(git ls-files -s)

# 3. FetchContent commit pins in tracked CMake files.
while IFS= read -r cmake_file; do
  while IFS= read -r sha; do
    allowed_shas+=("${sha}")
    notice_has "${sha}" || err "${cmake_file} pins GIT_TAG ${sha}, which NOTICE does not record"
  done < <(sed -n 's/^[[:space:]]*GIT_TAG[[:space:]]\{1,\}\([0-9a-f]\{40\}\).*/\1/p' "${cmake_file}")
done < <(git ls-files -- '*CMakeLists.txt' '*.cmake')

# 4. Every full commit SHA in NOTICE is still pinned.
while IFS= read -r sha; do
  found=0
  for pinned in ${allowed_shas[@]+"${allowed_shas[@]}"}; do
    [[ "${pinned}" == "${sha}" ]] && { found=1; break; }
  done
  [[ ${found} -eq 1 ]] || err "NOTICE records commit ${sha}, which no submodule or FetchContent pin uses (stale hash)"
done < <(grep -oE '[0-9a-f]{40,}' NOTICE | awk 'length($0) == 40' | sort -u)

# 5. No unattributed binaries or data files.
data_ext='\.(bin|gguf|ggml|safetensors|ckpt|pt|pth|onnx|tflite|mlmodel|mlpackage|npy|npz|wav|pcm|raw|f32|mp3|flac|ogg|m4a|png|jpe?g|gif|webp|bmp|ico|icns|zip|jar|aar|tar|gz|tgz|xz|7z)$'
empty_tree="$(git hash-object -t tree /dev/null)"
binaries="$(git diff --numstat "${empty_tree}" HEAD 2>/dev/null | awk -F'\t' '$1=="-"{print $3}' || true)"
# Also staged / untracked-but-added files that are not yet committed.
binaries+=$'\n'"$(git diff --cached --numstat "${empty_tree}" 2>/dev/null | awk -F'\t' '$1=="-"{print $3}' || true)"
while IFS= read -r path; do
  [[ -n "${path}" ]] || continue
  git ls-files --error-unmatch -- "${path}" >/dev/null 2>&1 || continue   # deleted since
  [[ "$(git ls-files -s -- "${path}" | awk '{print $1}')" == 160000 ]] && continue
  notice_has "\`${path}\`" || err "tracked binary/data file ${path} is not recorded in NOTICE"
done < <({ printf '%s\n' "${binaries}"; git ls-files | grep -iE "${data_ext}" || true; } | sort -u)

# 6. Fixture table rows.
fixture_rows="$(awk '/^## Test Fixtures/{in_section=1; next} /^## /{in_section=0} in_section && /^\| /' NOTICE \
  | grep -vE '^\| (Fixture Path|---)' || true)"
if [[ -z "${fixture_rows}" ]]; then
  err "NOTICE has no '## Test Fixtures' table rows"
fi
while IFS= read -r row; do
  [[ -n "${row}" ]] || continue
  IFS='|' read -r _ fixture source license _notes <<<"${row}"
  license="$(echo "${license}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
  if ! [[ "${license}" =~ ^(MIT|CC0)([^A-Za-z0-9-]|-1\.0|$) ]]; then
    err "fixture row has a license that is not MIT or CC0 (${license}): ${fixture}"
  fi
  if [[ "${fixture}" != *"generated at test time"* ]]; then
    while IFS= read -r path; do
      git ls-files --error-unmatch -- "${path}" >/dev/null 2>&1 || err "committed fixture ${path} in NOTICE is not tracked"
    done < <(grep -oE '`[^`]+`' <<<"${fixture}" | tr -d '`')
  fi
  # Generator sources: backticked repo-relative paths with a file extension.
  while IFS= read -r path; do
    [[ -n "${path}" ]] || continue
    git ls-files --error-unmatch -- "${path}" >/dev/null 2>&1 || err "fixture source ${path} in NOTICE does not exist"
  done < <(grep -oE '`[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+`' <<<"${source}" | tr -d '`')
done <<<"${fixture_rows}"

if [[ ${errors} -gt 0 ]]; then
  echo "license audit: ${errors} problem(s) found" >&2
  exit 1
fi
echo "license audit: OK (LICENSE is MIT; $(git ls-files -s | awk '$1==160000' | wc -l | tr -d ' ') submodule pins," \
  "$(grep -c '' <<<"${fixture_rows}") fixture rows and all tracked binaries recorded in NOTICE)"
