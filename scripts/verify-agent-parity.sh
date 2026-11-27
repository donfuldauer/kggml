#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=ci/python.sh
source "${REPO_ROOT}/scripts/ci/python.sh"
kggml_require_python 3.11

errors=0

check_identical() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$dst" ]]; then
    echo "ERROR: Missing mirror file: $dst (expected mirror of $src)" >&2
    errors=$((errors + 1))
    return
  fi
  if ! cmp -s "$src" "$dst"; then
    echo "ERROR: Mirror drift detected between $src and $dst" >&2
    errors=$((errors + 1))
  fi
}

# 1. Verify root instructions mirrors
for target in CLAUDE.md GEMINI.md .gemini/GEMINI.md; do
  check_identical "AGENTS.md" "$target"
done

# 2. Verify canonical agent markdown mirrors and no stray files
canonical_agents=()
for f in .agents/agents/*.md; do
  canonical_agents+=("$(basename "$f")")
done

for dir in .claude/agents .gemini/agents .antigravity/agents .codex/agents; do
  if [[ ! -d "$dir" ]]; then
    echo "ERROR: Missing mirror directory: $dir" >&2
    errors=$((errors + 1))
    continue
  fi
  for base in "${canonical_agents[@]}"; do
    check_identical ".agents/agents/$base" "$dir/$base"
  done
  for existing in "$dir"/*.md; do
    [[ -e "$existing" ]] || continue
    base="$(basename "$existing")"
    if [[ ! -f ".agents/agents/$base" ]]; then
      echo "ERROR: Stray agent mirror not present in .agents/agents/: $existing" >&2
      errors=$((errors + 1))
    fi
  done
done

# 3. Verify canonical skills directory mirrors
for dir in .claude/skills .gemini/skills .antigravity/skills; do
  if [[ ! -d "$dir" ]]; then
    echo "ERROR: Missing skills mirror directory: $dir" >&2
    errors=$((errors + 1))
    continue
  fi
  if ! diff -qr ".agents/skills" "$dir" >/dev/null; then
    echo "ERROR: Skills directory drift detected between .agents/skills and $dir" >&2
    diff -qr ".agents/skills" "$dir" >&2 || true
    errors=$((errors + 1))
  fi
done

# 4. Verify .codex/config.toml and .codex/agents/*.toml parse cleanly and match canonical agents
kggml_python - <<'PY' || errors=$((errors + 1))
import glob
import os
import re
import sys
import tomllib

READ_ONLY_AGENTS = {"issue-triager", "phase-reviewer"}

def parse_frontmatter(text: str):
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        raise ValueError("Missing YAML frontmatter")
    fm_raw, body = match.group(1), match.group(2).lstrip("\n")
    meta = {}
    for line in fm_raw.splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body

with open(".codex/config.toml", "rb") as f:
    cfg = tomllib.load(f)

agents_cfg = cfg.get("agents", {})
canonical_names = set()

for path in sorted(glob.glob(".agents/agents/*.md")):
    with open(path, "r", encoding="utf-8") as f:
        meta, body = parse_frontmatter(f.read())
    name = meta["name"]
    description = meta["description"]
    canonical_names.add(name)

    if name not in agents_cfg:
        print(f"ERROR: Missing [agents.\"{name}\"] entry in .codex/config.toml", file=sys.stderr)
        sys.exit(1)

    entry = agents_cfg[name]
    if entry.get("description") != description:
        print(f"ERROR: Description mismatch for '{name}' in .codex/config.toml", file=sys.stderr)
        sys.exit(1)

    expected_rel = f"agents/{name}.toml"
    if entry.get("config_file") != expected_rel:
        print(f"ERROR: config_file mismatch for '{name}': got {entry.get('config_file')}, expected {expected_rel}", file=sys.stderr)
        sys.exit(1)

    role_path = os.path.join(".codex", expected_rel)
    if not os.path.isfile(role_path):
        print(f"ERROR: Missing Codex role file {role_path}", file=sys.stderr)
        sys.exit(1)

    with open(role_path, "rb") as rf:
        role_cfg = tomllib.load(rf)

    expected_sandbox = "read-only" if name in READ_ONLY_AGENTS else "workspace-write"
    if role_cfg.get("sandbox_mode") != expected_sandbox:
        print(f"ERROR: sandbox_mode mismatch in {role_path}", file=sys.stderr)
        sys.exit(1)

    if role_cfg.get("developer_instructions") != body:
        print(f"ERROR: developer_instructions drift in {role_path}", file=sys.stderr)
        sys.exit(1)

# Check for stray .toml role files
for toml_path in sorted(glob.glob(".codex/agents/*.toml")):
    stem = os.path.splitext(os.path.basename(toml_path))[0]
    if stem not in canonical_names:
        print(f"ERROR: Stray Codex role TOML file: {toml_path}", file=sys.stderr)
        sys.exit(1)
PY

if [[ "$errors" -gt 0 ]]; then
  echo "Agent parity verification FAILED with $errors error(s). Run ./scripts/sync-agent-mirrors.sh to sync." >&2
  exit 1
fi

echo "Agent parity verification PASSED across AGENTS.md, .agents, .claude, .gemini, .antigravity, and .codex."
