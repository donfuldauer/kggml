#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=ci/python.sh
source "${REPO_ROOT}/scripts/ci/python.sh"
kggml_require_python 3.11

# 1. Mirror root AGENTS.md to CLAUDE.md, GEMINI.md, and .gemini/GEMINI.md
mkdir -p .gemini
cp AGENTS.md CLAUDE.md
cp AGENTS.md GEMINI.md
cp AGENTS.md .gemini/GEMINI.md

# 2. Mirror canonical .agents/agents/*.md to .claude, .gemini, .antigravity, and .codex
for dir in .claude/agents .gemini/agents .antigravity/agents .codex/agents; do
  mkdir -p "$dir"
  rm -f "$dir"/*.md
done
rm -f .codex/agents/*.toml

for agent_file in .agents/agents/*.md; do
  base="$(basename "$agent_file")"
  cp "$agent_file" ".claude/agents/$base"
  cp "$agent_file" ".gemini/agents/$base"
  cp "$agent_file" ".antigravity/agents/$base"
  cp "$agent_file" ".codex/agents/$base"
done

# 3. Mirror canonical .agents/skills/ to .claude/skills, .gemini/skills, and .antigravity/skills
for dir in .claude/skills .gemini/skills .antigravity/skills; do
  rm -rf "$dir"
  mkdir -p "$dir"
  cp -R .agents/skills/. "$dir/"
done

# 4. Generate .codex/config.toml and .codex/agents/<name>.toml from canonical .agents/agents/*.md
kggml_python - <<'PY'
import glob
import os
import re

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

def toml_escape_basic(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def toml_multiline(s: str) -> str:
    # Escape backslashes and any triple double-quotes safely for TOML basic multiline strings
    escaped = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'"""\n{escaped}"""'

agents = []
for path in sorted(glob.glob(".agents/agents/*.md")):
    with open(path, "r", encoding="utf-8") as f:
        meta, body = parse_frontmatter(f.read())
    name = meta["name"]
    description = meta["description"]
    sandbox = "read-only" if name in READ_ONLY_AGENTS else "workspace-write"
    agents.append((name, description, sandbox, body))

    role_toml_path = os.path.join(".codex", "agents", f"{name}.toml")
    with open(role_toml_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(f"# Auto-generated from .agents/agents/{name}.md by scripts/sync-agent-mirrors.sh\n")
        out.write(f'sandbox_mode = "{sandbox}"\n')
        out.write(f"developer_instructions = {toml_multiline(body)}\n")

config_lines = [
    "# Auto-generated from .agents/agents/*.md by scripts/sync-agent-mirrors.sh",
    "# OpenAI Codex multi-agent role registry for Kggml",
    "",
    "[agents]",
    "max_threads = 4",
    "max_depth = 2",
    "",
]
for name, description, _sandbox, _body in agents:
    config_lines.extend([
        f'[agents."{name}"]',
        f'description = "{toml_escape_basic(description)}"',
        f'config_file = "agents/{name}.toml"',
        "",
    ])

with open(os.path.join(".codex", "config.toml"), "w", encoding="utf-8", newline="\n") as out:
    out.write("\n".join(config_lines))
PY

echo "Synced all agent instructions, personas, skills, and .codex TOML configs."
