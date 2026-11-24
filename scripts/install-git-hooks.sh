#!/usr/bin/env bash
# Points this clone at the tracked hooks in .githooks/ (idempotent, quiet).
# Run once per clone; agent harnesses run it automatically at session start
# (.claude/settings.json SessionStart hook) and ./scripts/ci-local.sh runs it too.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$(git config --get core.hooksPath || true)" != ".githooks" ]]; then
  git config core.hooksPath .githooks
  echo "git hooks: core.hooksPath set to .githooks"
fi
