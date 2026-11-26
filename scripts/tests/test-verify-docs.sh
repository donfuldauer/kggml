#!/usr/bin/env bash
# Tests for scripts/ci/verify-docs.sh against small throwaway git repositories.
#
# Asserts the audit passes for a clean repository (and for this repository), and FAILS for: a broken relative link,
# a broken heading anchor, a local machine path and a TODO in a user-facing document, an unpublished artifact id, a
# wrong BOM version, a README Kotlin block without a snippet marker, a README block that drifted from its compiled
# snippet, an unused snippet region, and public declarations (function, property, constructor property) without
# KDoc.
#
# Usage: scripts/tests/test-verify-docs.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AUDIT="${REPO_ROOT}/scripts/ci/verify-docs.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

passed=0
failed=0
ok() { echo "  ok   $1"; passed=$((passed + 1)); }
ko() { echo "  FAIL $1" >&2; failed=$((failed + 1)); }

# make_repo <dir>: a clean repository the audit accepts.
make_repo() {
  local dir="$1"
  mkdir -p "${dir}/kggml-bom" "${dir}/docs" "${dir}/docs-snippets/src" "${dir}/kggml-demo/src/commonMain/kotlin"
  git -C "${dir}" init -q
  printf 'GROUP=com.donfuldauer\nVERSION_NAME=1.2.3\n' >"${dir}/gradle.properties"
  cat >"${dir}/kggml-bom/build.gradle.kts" <<'EOF'
val kggmlLibraryModules: List<String> = listOf(
    ":kggml-core-api",
    ":kggml-demo",
)
EOF
  cat >"${dir}/README.md" <<'EOF'
# Demo

See [the guide](docs/GUIDE.md#getting-started), [the API section](#api) and <https://example.invalid/>.

## API

<!-- snippet: hello -->
```kotlin
implementation(platform("com.donfuldauer:kggml-bom:1.2.3"))
println("hello")
```

```bash
echo "shell blocks are not snippets"
```
EOF
  cat >"${dir}/docs/GUIDE.md" <<'EOF'
# Guide

## Getting started

Back to the [README](../README.md).
EOF
  cat >"${dir}/docs-snippets/src/Hello.kt" <<'EOF'
fun main() {
    // snippet:start hello
    implementation(platform("com.donfuldauer:kggml-bom:1.2.3"))
    println("hello")
    // snippet:end hello
}
EOF
  cat >"${dir}/kggml-demo/src/commonMain/kotlin/Demo.kt" <<'EOF'
package demo

/**
 * A demo.
 *
 * @property name the name.
 * @property size the size.
 */
public data class Demo(
    val name: String,
    public val size: Int,
) {
    /** Says hello. */
    public fun hello(): String = "hi"

    public companion object {
        /** The default. */
        public val DEFAULT: Demo = Demo("d", 1)
    }

    override fun toString(): String = name
}

/** Documented. */
@Suppress("unused")
public fun topLevel(): Int = 1

internal fun notPublic(): Int = 2
EOF
  git -C "${dir}" add -A
}

# expect <name> <pass|fail> <dir> [pattern]: run the audit on <dir>; for failures, also require <pattern> in stderr.
expect() {
  local name="$1" want="$2" dir="$3" pattern="${4:-}" out rc=0
  out="$(KGGML_DOCS_AUDIT_ROOT="${dir}" "${AUDIT}" 2>&1)" || rc=$?
  if [[ "${want}" == pass ]]; then
    if [[ ${rc} -eq 0 ]]; then ok "${name}"; else ko "${name}: expected pass, got:"; echo "${out}" >&2; fi
    return
  fi
  if [[ ${rc} -eq 0 ]]; then ko "${name}: expected failure, audit passed"; return; fi
  if [[ -n "${pattern}" ]] && ! grep -qE -- "${pattern}" <<<"${out}"; then
    ko "${name}: failed without '${pattern}':"; echo "${out}" >&2; return
  fi
  ok "${name}"
}

# case_repo <name>: a fresh clean repository for one failure case.
case_repo() {
  local dir="${WORK}/$1"
  make_repo "${dir}"
  echo "${dir}"
}

echo "verify-docs.sh tests"

expect "clean repository passes" pass "$(case_repo clean)"

d="$(case_repo broken-link)"
sed -i.bak 's|docs/GUIDE.md#getting-started|docs/MISSING.md|' "${d}/README.md" && git -C "${d}" add -A
expect "broken relative link fails" fail "${d}" "broken link 'docs/MISSING.md'"

d="$(case_repo broken-anchor)"
sed -i.bak 's|#getting-started|#getting-going|' "${d}/README.md" && git -C "${d}" add -A
expect "broken anchor fails" fail "${d}" "broken anchor '#getting-going'"

d="$(case_repo local-path)"
echo "Built in /home/someone/kggml." >>"${d}/README.md" && git -C "${d}" add -A
expect "local machine path fails" fail "${d}" "absolute local machine path"

d="$(case_repo todo)"
echo "TODO: write this section." >>"${d}/README.md" && git -C "${d}" add -A
expect "TODO placeholder fails" fail "${d}" "unfinished placeholder"

d="$(case_repo unknown-artifact)"
echo 'Add `com.donfuldauer:kggml-server` to your build.' >>"${d}/README.md" && git -C "${d}" add -A
expect "unpublished artifact fails" fail "${d}" "kggml-server is not a published artifact"

d="$(case_repo wrong-version)"
sed -i.bak 's/1\.2\.3/9.9.9/' "${d}/README.md" "${d}/docs-snippets/src/Hello.kt" && git -C "${d}" add -A
expect "wrong BOM version fails" fail "${d}" "does not match VERSION_NAME=1.2.3"

d="$(case_repo no-marker)"
printf '\n```kotlin\nprintln("unchecked")\n```\n' >>"${d}/README.md" && git -C "${d}" add -A
expect "README block without snippet marker fails" fail "${d}" "has no <!-- snippet"

d="$(case_repo drift)"
sed -i.bak 's/println("hello")/println("hello, world")/' "${d}/README.md" && git -C "${d}" add -A
expect "README block that drifted from its snippet fails" fail "${d}" "differs from snippet 'hello'"

d="$(case_repo unused)"
printf '// snippet:start orphan\nval x = 1\n// snippet:end orphan\n' >>"${d}/docs-snippets/src/Hello.kt" && git -C "${d}" add -A
expect "unused snippet region fails" fail "${d}" "snippet 'orphan' is not used"

d="$(case_repo undocumented-fun)"
sed -i.bak 's|^/\*\* Documented. \*/$||' "${d}/kggml-demo/src/commonMain/kotlin/Demo.kt" && git -C "${d}" add -A
expect "public function without KDoc fails" fail "${d}" "public fun 'topLevel' has no KDoc"

d="$(case_repo undocumented-property)"
sed -i.bak 's|^        /\*\* The default. \*/$||' "${d}/kggml-demo/src/commonMain/kotlin/Demo.kt" && git -C "${d}" add -A
expect "public property without KDoc fails" fail "${d}" "public property 'DEFAULT' has no KDoc"

d="$(case_repo undocumented-ctor-property)"
sed -i.bak '/@property size the size./d' "${d}/kggml-demo/src/commonMain/kotlin/Demo.kt" && git -C "${d}" add -A
expect "constructor property without @property fails" fail "${d}" "public property 'size' has no KDoc"

expect "this repository passes" pass "${REPO_ROOT}"

echo "${passed} passed, ${failed} failed"
[[ ${failed} -eq 0 ]]
