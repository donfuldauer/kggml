#!/usr/bin/env bash
# Sourced helper: sets KGGML_PYTHON to a Python 3 interpreter that is at least the requested version
# (default 3.11, the first release with `tomllib`), or exits with an error.
#
# Why: macOS ships /usr/bin/python3 3.9 ahead of any Homebrew python on a non-login PATH, and Windows
# (Git Bash on a self-hosted runner) often has only `python` or the `py` launcher, not `python3`.
#
# Usage:
#   source "${REPO_ROOT}/scripts/ci/python.sh"
#   kggml_require_python 3.11
#   kggml_python - <<'PY' ... PY       (or "${KGGML_PYTHON}" when it is a single executable)
# KGGML_PYTHON may be preset by the caller (e.g. CI's setup-python step); it is then only version-checked.

kggml_python_ok() {
  local interpreter="$1" major="$2" minor="$3"
  "${interpreter}" -c "import sys; sys.exit(0 if sys.version_info >= (${major}, ${minor}) else 1)" >/dev/null 2>&1
}

kggml_require_python() {
  local wanted="${1:-3.11}" major minor candidate
  major="${wanted%%.*}"
  minor="${wanted#*.}"
  if [[ -n "${KGGML_PYTHON:-}" ]]; then
    if kggml_python_ok "${KGGML_PYTHON}" "${major}" "${minor}"; then return 0; fi
    echo "ERROR: KGGML_PYTHON=${KGGML_PYTHON} is older than Python ${wanted}" >&2
    return 1
  fi
  for candidate in python3 python3.14 python3.13 python3.12 python3.11 python \
      /opt/homebrew/bin/python3 /usr/local/bin/python3; do
    command -v "${candidate}" >/dev/null 2>&1 || [[ -x "${candidate}" ]] || continue
    if kggml_python_ok "${candidate}" "${major}" "${minor}"; then
      KGGML_PYTHON="${candidate}"
      export KGGML_PYTHON
      return 0
    fi
  done
  # Windows Python launcher.
  if command -v py >/dev/null 2>&1 && py -3 -c "import sys; sys.exit(0 if sys.version_info >= (${major}, ${minor}) else 1)" >/dev/null 2>&1; then
    KGGML_PYTHON="py"
    KGGML_PYTHON_LAUNCHER_ARGS="-3"
    export KGGML_PYTHON KGGML_PYTHON_LAUNCHER_ARGS
    return 0
  fi
  echo "ERROR: Python >= ${wanted} not found (tried python3, python3.1x, python, py -3). Install it or set KGGML_PYTHON." >&2
  return 1
}

# Runs the interpreter chosen by kggml_require_python with the given arguments.
kggml_python() {
  # shellcheck disable=SC2086 # KGGML_PYTHON_LAUNCHER_ARGS is empty or a single word.
  "${KGGML_PYTHON:?call kggml_require_python first}" ${KGGML_PYTHON_LAUNCHER_ARGS:-} "$@"
}
