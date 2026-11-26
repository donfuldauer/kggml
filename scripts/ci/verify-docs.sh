#!/usr/bin/env bash
# Documentation audit (issue #22). Fails if:
#   1. a relative link or #anchor in any tracked Markdown file points at a file, directory or heading that does not
#      exist (external http(s) links are only syntax-checked unless --online is given);
#   2. a user-facing document (README, CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, NOTICE, .github templates, module and
#      sample READMEs) contains an internal-only reference: a local machine path, a private agent-session link, or a
#      TODO / TBD / FIXME placeholder;
#   3. a `com.donfuldauer:<artifact>[:<version>]` coordinate in those documents or in docs-snippets/ names an artifact
#      that is not published (kggml-bom + the BOM's library list) or a version other than VERSION_NAME;
#   4. a Kotlin code block in README.md is not preceded by `<!-- snippet: <name> -->`, or differs from the region
#      `// snippet:start <name>` ... `// snippet:end <name>` in docs-snippets/ (which :docs-snippets compiles against
#      the published artifacts), or a region is unused;
#   5. a public declaration in a library module's main source sets has no KDoc (its own `/** */`, or an `@property` /
#      `@param` line in the KDoc of its class).
# Reads only tracked files. Shared by .github/workflows/repo-hygiene.yml and ./scripts/ci-local.sh.
# KGGML_DOCS_AUDIT_ROOT overrides the repository root (used by its tests).
#
# Usage: scripts/ci/verify-docs.sh [--online] [--only links,internal,coordinates,snippets,kdoc]
set -euo pipefail

exec python3 -B "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verify_docs.py" "$@"
