---
name: update-dependencies
description: Autonomous end-to-end dependency update workflow for Kggml. Audits and updates Gradle catalog dependencies, Gradle wrapper, CMake FetchContent pins, GitHub Actions, pinned CI tools, and compatible native submodules following stable/pre-stable version rules, verifies locally, opens a PR, waits for CI, and merges autonomously across Antigravity and Claude Code.
---

# Skill: `update-dependencies`

Audit and update all dependencies across Kggml, verify locally, open a pull request, wait for CI to pass, and merge into `main` with a merge commit using standard CLI tools (`gh`, `git`, `./gradlew`, `cmake`, `ctest`, `python3`, `curl`).

## Trigger Forms

- `"update all dependencies"` / `"update dependencies"` / `"bump dependencies"`
- Optional qualifiers such as opening and merging a PR once ready.

## Standing Authorization (all harnesses)

Invoking this skill (or asking to update dependencies and open/merge the PR) authorizes the agent to run the **entire** workflow end to end across **Antigravity**, **Claude Code**, and other harnesses: cutting a branch from `main`, updating dependencies, verifying locally, pushing, opening the PR, waiting for CI, fixing CI issues if needed, and **merging into `main` with a merge commit** without stopping for confirmation.

Stop and report to the user instead of merging **only** when:
- a required CI check fails with a code error that cannot be resolved cleanly (runner failures/hangs or platform-only code failures after two fix pushes follow `AGENTS.md` "When a runner fails or hangs");
- the PR has a merge conflict that cannot be resolved cleanly; or
- a human reviewer requests changes that remain unresolved.

---

## Version Upgrade Policy

Apply these rules strictly to every dependency:

1. **Stable versions** (no pre-release qualifier such as `-alpha`, `-beta`, `-rc`, `-M`, `-SNAPSHOT`, `-dev`):
   - Update to the **latest stable** version.
   - Never upgrade a stable dependency to a pre-stable version.
2. **Pre-stable versions** (`alpha`, `beta`, `rc`, etc.):
   - Order maturity as `alpha < beta < rc < stable`.
   - If a **newer stable** version exists (higher semver than the pre-stable base, or the stable release of that same version), update to the **latest stable** version.
   - Otherwise, update to the **latest pre-stable** version whose maturity rank is **at least** the current maturity rank (e.g. `alpha` -> newer `alpha`/`beta`/`rc`/`stable`; `beta` -> newer `beta`/`rc`/`stable`; `rc` -> newer `rc`/`stable`).
   - Never regress maturity (e.g. never move `beta` -> `alpha` or `rc` -> `beta`/`alpha`, even for a higher minor/major version).
3. **Repository invariants that override blind major bumps:**
   - `robolectric-android-all` in `gradle/libs.versions.toml` must stay on the latest build matching `sdk=<N>` in the samples' `robolectric.properties` (e.g. `14-robolectric-*` while `sdk=34` is used for JDK 17 compatibility).
   - Upstream C/C++ submodules in `NOTICE` and `.gitmodules` must stay ABI-compatible with `:kggml-runtime`'s `ggml` (in particular, `llama.cpp` must match the runtime's `ggml` API/ABI, and `stable-diffusion.cpp` must target upstream `ggml-org/ggml` rather than an incompatible fork).
   - Compose Multiplatform, Compose Material3, and `androidx-compose-ui` must remain mutually compatible and pass Android Robolectric, desktop JVM, and iOS/macOS builds; bump `android-compileSdk` / `android-compileSdkMinor` only if required by an updated AndroidX/Compose artifact.

---

## Step 1: Sync `main` & Create a Branch

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git checkout -b chore/update-dependencies
./scripts/install-git-hooks.sh
```

If `chore/update-dependencies` already exists, use `chore/update-dependencies-<YYYY-MM-DD>`.

---

## Step 2: Inventory & Query Latest Versions

Check **every** dependency surface in the repository:

1. **Gradle Version Catalog (`gradle/libs.versions.toml`)** and any hardcoded coordinates in `*.gradle.kts`:
   - Query `maven-metadata.xml` from:
     - Google Maven: `https://dl.google.com/dl/android/maven2/<group-path>/<artifact>/maven-metadata.xml` (`com.android.*`, `com.google.*`, `androidx.*`)
     - Maven Central: `https://repo1.maven.org/maven2/<group-path>/<artifact>/maven-metadata.xml` (if `repo1.maven.org` returns HTTP 429 on shared cloud egress IPs, use Google's official Maven Central mirror `https://maven-central.storage-download.googleapis.com/maven2/<group-path>/<artifact>/maven-metadata.xml` and cross-check against GitHub releases via `gh api repos/<owner>/<repo>/releases`)
     - Gradle Plugin Portal: `https://plugins.gradle.org/m2/<group-path>/<artifact>/maven-metadata.xml`
   - Also check `rg -n '"[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+:[0-9]' --glob '*.gradle.kts' .` for any versioned coordinates outside `libs.versions.toml` (excluding Kggml's own BOM in `docs-snippets/`).

2. **Gradle Wrapper (`gradle/wrapper/gradle-wrapper.properties`):**
   - Query `https://services.gradle.org/versions/current` for the latest stable Gradle release.
   - If newer than `distributionUrl`, run `./gradlew wrapper --gradle-version <VERSION> --distribution-type bin` **twice** (so the new Gradle version regenerates `gradle-wrapper.jar`, `gradlew`, and `gradlew.bat`), and update the `Gradle Wrapper` row in `NOTICE`.

3. **CMake `FetchContent` Pins (`kggml-*/native/tests/CMakeLists.txt`):**
   - Check every `FetchContent_Declare` / `GIT_TAG` (currently GoogleTest across `kggml-runtime`, `kggml-whisper`, `kggml-llama`, and `kggml-diffusion`).
   - Query latest stable release tag and commit SHA (`gh api repos/google/googletest/releases` and `gh api repos/google/googletest/git/ref/tags/<tag>`).
   - Keep all four `native/tests/CMakeLists.txt` files on the exact same commit SHA, and update `NOTICE` and `kggml-runtime/native/README.md`.

4. **GitHub Actions & Pinned CI Tools (`.github/workflows/*.yml`):**
   - Check every `uses:` action via `gh api repos/<owner>/<repo>/releases` and update major version tags (`@vN`) when a newer stable major release exists.
   - Check any pinned CLI binaries in workflows (such as `GITLEAKS_VERSION` / `GITLEAKS_SHA256` in `.github/workflows/repo-hygiene.yml`), verifying the release checksum from the upstream checksums file.

5. **Native C/C++ Git Submodules (`.gitmodules`, `NOTICE`, `kggml-*/native/CMakeLists.txt`):**
   - Check upstream releases/HEADs for `ggml`, `whisper.cpp`, `llama.cpp`, `stable-diffusion.cpp`, `Vulkan-Headers`, `SPIRV-Headers`, `OpenCL-Headers`, and `OpenCL-ICD-Loader`.
   - Only bump submodules when an ABI-compatible newer version exists; when updating any submodule, also update its pin in `CMakeLists.txt`, `NOTICE`, module `native/README.md`, and verify `scripts/ci/verify-*-pin.sh` and `scripts/ci/verify-licenses.sh`.

If every dependency is already at its target version, report that no updates are needed and stop without creating an empty PR.

---

## Step 3: Apply Updates & Verify Locally

1. Apply all eligible updates and keep `NOTICE` and docs in sync (`scripts/ci/verify-licenses.sh` and `scripts/ci/verify-docs.sh` enforce exact parity).
2. If you touched `AGENTS.md`, `.agents/agents/`, or `.agents/skills/`, run:
   ```bash
   ./scripts/sync-agent-mirrors.sh
   ./scripts/verify-agent-parity.sh
   ```
3. Initialize submodules if native paths changed (including `gradle/`, `NOTICE`, or `kggml-*/native/`):
   ```bash
   git submodule update --init --depth 1
   ```
4. Run the local verification gates before pushing (fix every `FAIL` locally):
   ```bash
   ./gradlew help --warning-mode all --console=plain   # resolve any newly deprecated Gradle APIs in our build scripts
   ./scripts/ci-local.sh
   ./gradlew check --console=plain
   ```
   - Note: on environments where `repo.maven.apache.org` rate-limits (HTTP 429), configure a user-local Gradle init script under `~/.gradle/init.d/` (never committed to the repository) to route Maven Central through `https://maven-central.storage-download.googleapis.com/maven2`.
   - Save the `ci-local.sh` summary block for the PR body.

---

## Step 4: Commit, Push & Open PR

1. Stage and commit with proper attribution (do not bypass `.githooks/commit-msg` with `--no-verify`):
   ```bash
   ./scripts/install-git-hooks.sh
   git status
   git add -A
   git commit -m "build(deps): update dependencies to latest compatible versions"
   ```
   If the git author is the human contributor rather than an agent identity, add an explicit agent `Co-authored-by:` trailer so both the human and agent are credited per `AGENTS.md`.
2. Push and open the PR with a clear before/after dependency table and the `ci-local.sh` verification summary:
   ```bash
   git push -u origin "$(git branch --show-current)"
   gh pr create \
     --base main \
     --head "$(git branch --show-current)" \
     --title "build(deps): update dependencies to latest compatible versions" \
     --body "## Summary\n..."
   ```

---

## Step 5: Wait for CI & Auto-Fix (Antigravity & Claude Code)

Never end a turn while a check on the PR head is still pending (see `AGENTS.md` "Waiting for CI").

- **Claude Code:**
  1. Immediately call `subscribe_pr_activity` with `owner`, `repo`, and `pullNumber=<PR_NUMBER>`.
  2. Start `./scripts/ci/wait-pr-checks.sh <PR_NUMBER>` via Bash with `run_in_background: true` (the harness re-invokes you as soon as the waiter exits).
- **Antigravity:**
  1. Run `./scripts/ci/wait-pr-checks.sh <PR_NUMBER>` with `run_command` (with `NotificationTimeoutSeconds` set, letting it run as a tracked background task until the system notifies you on completion — never poll in a loop).
- **Handle the waiter exit code:**
  - `0` (all checks green): proceed immediately to Step 6.
  - `1` (failure or cancellation): inspect failing job logs (`gh run view <RUN_ID> --log-failed`). Fix code failures locally, re-run `./scripts/ci-local.sh`, push a batched fix commit, and restart `./scripts/ci/wait-pr-checks.sh <PR_NUMBER>`. For infrastructure failures or macOS/Windows-only failures after two fix pushes, follow `AGENTS.md` "When a runner fails or hangs".
  - `2` (queued 20 min / running 60 min deadline): follow `AGENTS.md` "When a runner fails or hangs".
  - `3` (`gh` cannot query checks): poll check runs via GitHub MCP `get_check_runs` every 90 s using the same deadlines.

---

## Step 6: Merge Autonomously

Once all required checks pass (or the `AGENTS.md` runner-failure local gate applies) and there are no conflicts or unresolved requested changes:

```bash
gh pr checks <PR_NUMBER>
gh pr merge <PR_NUMBER> --merge
git checkout main
git pull --ff-only origin main
git branch -D chore/update-dependencies 2>/dev/null || true
```

- Always use `--merge` (squash merges are disabled on the repository).
- Never pass `--delete-branch` or delete the remote branch manually (GitHub deletes head branches automatically on merge).
- On Claude Code, call `unsubscribe_pr_activity` after the PR merges.
- Report the final dependency table, PR URL, and merge commit SHA to the user.
