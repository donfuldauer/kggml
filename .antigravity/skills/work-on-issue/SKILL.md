---
name: work-on-issue
description: Autonomous end-to-end GitHub issue-to-PR workflow for Kggml. Handles "work on next issue" or "work on issue #X" using standard gh, git, ./gradlew, and ctest CLI commands across Claude Code, Gemini/Antigravity, and OpenAI Codex.
---

# Skill: `work-on-issue`

Execute a Kggml GitHub issue from selection through implementation, verification, PR review/phase-gate, merge-commit merge, and downstream DAG unblocking using strictly standard CLI tools (`gh`, `git`, `./gradlew`, `ctest`).

## Trigger Forms

- `"work on next issue"` — select the highest-priority open issue labeled `status:ready` with zero unresolved `Blocked by` dependencies.
- `"work on issue #X"` — verify issue `#X` is open, labeled `status:ready`, and has all `Blocked by` prerequisites closed.

## Standing Authorization (all harnesses)

Invoking this skill is the user's explicit, standing authorization to run the **entire** workflow end to end, including pushing the feature branch, opening the PR, **merging it into `main` with a merge commit**, and relabeling/unblocking downstream issues. Do **not** stop to ask for confirmation before merging. The agent performs the merge itself as soon as every gate in Step 5 passes.

Stop and report to the user instead of merging **only** when:
- a required CI check (or, on a `ci:macos`/`ci:windows`-labelled PR, that check) is failing and cannot be fixed within the issue's scope (checks that never started because the Actions quota is exhausted, and runners that fail, hang or time out, are not failures to escalate: follow AGENTS.md "When the Actions quota is exhausted" and "When a runner fails or hangs");
- the PR has a merge conflict that cannot be resolved cleanly;
- a phase-final PR lacks `APPROVED` from `phase-reviewer`; or
- a human reviewer has requested changes that are still unresolved.

Never stop, wait, or leave a PR open for manual or physical-device testing (AGENTS.md Section 3, "Deferred Manual & Physical Verification"). Those checks are deferred to the owner's final acceptance pass after all phases: document the procedure, add a line to `docs/MANUAL_VERIFICATION.md`, and merge on the automated gates.

---

## Step 1: Issue Selection & Atomic Claiming

1. **Select the target issue:**
   ```bash
   # For "work on next issue": list ready issues ordered by phase priority then issue number
   gh issue list --state open --label "status:ready" --limit 50 \
     --json number,title,labels,milestone,assignees
   ```
   Pick the lowest-priority-phase (`phase-1` < `phase-2` < `phase-3` < `phase-4` < `phase-oss`), lowest issue number issue with zero open `Blocked by #Y` dependencies.

2. **Verify dependencies:**
   ```bash
   gh issue view <N> --json number,title,body,state,labels,milestone
   ```
   Parse every `Blocked by #<DEP>` line in `## Dependencies` and confirm each `<DEP>` is `CLOSED`:
   ```bash
   gh issue view <DEP> --json state --jq .state
   ```
   If any blocker is still `OPEN`, stop and report the blocker without modifying labels.

3. **Claim the issue atomically:**
   ```bash
   gh issue edit <N> \
     --add-assignee @me \
     --remove-label "status:ready" \
     --add-label "status:in-progress"
   gh issue comment <N> --body "Claimed issue #<N>. Starting triage and implementation on branch \`feat/issue-<N>-<slug>\`."
   ```

---

## Step 2: Branching & Sub-Task Delegation

1. **Sync `main` and create the feature branch (`feat/issue-<N>-<slug>`):**
   ```bash
   git fetch origin
   git checkout main
   git pull --ff-only origin main
   git checkout -b feat/issue-<N>-<slug>
   ```
2. **Triage (`issue-triager`):**
   - Delegate to `.agents/agents/issue-triager.md` (or adopt the `issue-triager` role) to inspect the codebase, verify acceptance criteria, identify edge cases, and write an execution blueprint.
3. **Implement (`implementer`):**
   - Delegate to `.agents/agents/implementer.md` to implement all scoped deliverables adhering strictly to `AGENTS.md`.
4. **Test (`tester`):**
   - Delegate to `.agents/agents/tester.md` to author and run unit tests (`commonTest`), CTest/GoogleTest native suites, platform integration tests, binary audits, and Compose UI tests as required by the issue.

---

## Step 3: Local Verification & PR Creation

1. **Run verification gates locally (mandatory before every push, including auto-fix pushes):**
   ```bash
   # Runs the same checks as the Linux PR CI jobs (parity, gitleaks, ggml pin, host build + CTest,
   # linkage audit, jvmTest, Android cross-compile + audit). Must end with no FAIL lines.
   ./scripts/ci-local.sh
   # Plus any module-specific commands from AGENTS.md Section 3 that apply, e.g.:
   # ./gradlew check
   # ./gradlew iosSimulatorArm64Test   # macOS hosts only; otherwise covered by the ci:macos run
   ```
   Include `ci-local.sh`'s summary block in the PR's Verification section.
   If any acceptance criterion needs a person or a physical device, write its procedure (module or sample `README.md`) and add a `- [ ] [#<N>] <check> — <procedure link>` line to `docs/MANUAL_VERIFICATION.md` in this PR; in the PR body mark that item **Deferred to final acceptance**, not as a pending blocker. Do not push to see whether CI passes: CI minutes are limited, so fix failures locally first.
2. **Commit, push, and open PR:**
   ```bash
   git status
   git add -A
   ./scripts/install-git-hooks.sh   # idempotent; enables the contributor Co-authored-by hook
   git commit -m "<type>(<scope>): <concise summary> (#<N>)"
   git log -1 --format=%B | grep -iF "Co-authored-by: $(git config kggml.coauthor)"   # must match for agent-authored commits
   git push -u origin feat/issue-<N>-<slug>

   gh pr create \
     --base main \
     --head feat/issue-<N>-<slug> \
     --title "<concise PR title> (#<N>)" \
     --body "Closes #<N>\n\n## Summary\n...\n\n## Verification\n..."
   ```
3. **Request the macOS / Windows runs only when needed (AGENTS.md "CI Minutes Policy"):**
   Add the `ci:macos` label if the PR is phase-final (see Step 4) or changes macOS/iOS-specific native code (Metal, CoreML, Accelerate, `cinterop`, xcframework packaging, Apple-only `#ifdef`s). Add `ci:windows` if the PR is phase-final or changes Windows-specific code (MSVC flags, `windows-x64` presets, `_WIN32` `#ifdef`s). Otherwise leave them off.
   ```bash
   gh pr edit <PR_NUMBER> --add-label "ci:macos"
   gh pr edit <PR_NUMBER> --add-label "ci:windows"
   ```
4. **Enable auto-fix on the PR immediately (mandatory, do not ask first):**
   As soon as the PR exists, subscribe to its activity so CI failures, review comments, and review requests are delivered back to the agent and fixed autonomously on `feat/issue-<N>-<slug>`.
   - **Claude Code:** call the `subscribe_pr_activity` tool with `owner`, `repo`, and `pullNumber=<PR_NUMBER>` right after `gh pr create` (or the GitHub MCP `create_pull_request`) returns. This is part of opening the PR, not an optional follow-up.
   - **Other harnesses:** use the harness's equivalent PR watch / auto-fix mechanism.
   - **Wait for CI in the turn; never end a turn with a check pending** (AGENTS.md "Waiting for CI"). Subscriptions wake you only for failures, comments, reviews and the merge, and **not** when a check turns green. Right after every push or label change, start `scripts/ci/wait-pr-checks.sh <PR_NUMBER>` (Claude Code: Bash `run_in_background: true`, which re-invokes you when it exits) and act on its exit code:
     - `0` green: Step 5.
     - `1` failed: fix it, or apply the infrastructure rules.
     - `2` deadline: infrastructure failure.
     - `3` no `gh`: poll the GitHub MCP `get_check_runs` every 1–2 min, paced by a background `sleep 90`.
     Do not use 30-minute safety-net check-ins. If a turn has to end with checks pending, schedule one check-in at most 10 minutes out and confirm it exists (`CronList`).
   - Auto-fix pushes only to the PR's own feature branch; it never skips or disables tests to get green, and never force-pushes. Re-run `./scripts/ci-local.sh` before each auto-fix push and batch related fixes into a single push. Auto-fix only makes the PR mergeable; the merge itself happens in Step 5. Unsubscribe (`unsubscribe_pr_activity`) once the PR is merged or closed.
5. **Transition issue status to `status:in-review`:**
   ```bash
   gh issue edit <N> --remove-label "status:in-progress" --add-label "status:in-review"
   gh issue comment <N> --body "Opened PR #<PR_NUMBER> with implementation and verification results."
   ```

---

## Step 4: Conditional Phase Gate Check

Check whether `#N` is the **final open issue** in its milestone/phase:

```bash
gh issue list --state open --label "<phase-label>" --limit 50 --json number,title
```

- **Mid-phase PR (other issues in the phase remain open):** Proceed directly to Step 5 once CI checks pass. Do **not** invoke `phase-reviewer`.
- **Phase-final PR (only `#N` remains open in the phase):**
  1. Make sure the PR has the `ci:macos` and `ci:windows` labels so the macOS and Windows jobs run, then invoke `phase-reviewer` (`.agents/agents/phase-reviewer.md`) to audit cumulative phase diffs against Section 5 Exit Criteria.
  2. Fix any blocking findings, push updates, re-run verification, and obtain `APPROVED` from `phase-reviewer` before merging.

---

## Step 5: PR Merge & Downstream DAG Unblocking

Perform this step autonomously (see **Standing Authorization**): once the gates below pass, merge without asking the user. If CI is still running, wait for it with `scripts/ci/wait-pr-checks.sh <PR_NUMBER>` (or the MCP poll fallback) as in Step 3.4, up to the deadlines in AGENTS.md "When a runner fails or hangs". A missing event is never a reason to stop waiting; do not end the workflow with an open, green PR. Manual/physical checks recorded in `docs/MANUAL_VERIFICATION.md` are not merge gates.

1. **Verify CI checks pass and the PR is mergeable (no conflicts, no outstanding requested changes).** If the PR has the `ci:macos` / `ci:windows` label, the `ci.yml` macOS / Windows check must also be green:
   ```bash
   gh pr checks <PR_NUMBER>
   ```
   If checks never started because the Actions quota is exhausted (queued indefinitely, or failed instantly with a billing / spending-limit annotation and no logs), do not wait or re-push: follow AGENTS.md "When the Actions quota is exhausted" (`./scripts/ci-local.sh --full`, summary in the PR body, deferred macOS/Windows re-runs in `docs/MANUAL_VERIFICATION.md`) and merge on that local gate.

   If a check is queued past its deadline, lost its runner, was cancelled or timed out, or fails only on macOS/iOS or Windows after two fix pushes, follow AGENTS.md "When a runner fails or hangs": fall back in runner order once, then merge on `./scripts/ci-local.sh --full` with the deferred re-run (and, for a platform-only code failure, a follow-up issue) recorded. Never leave the PR waiting on a runner.
2. **Merge with a merge commit** (squash merges are disabled on the repository). Do not delete the remote branch or pass `--delete-branch`: the repository deletes head branches automatically on merge.
   ```bash
   gh pr merge <PR_NUMBER> --merge
   git checkout main
   git pull --ff-only origin main
   git branch -D feat/issue-<N>-<slug> 2>/dev/null || true
   ```
3. **Verify issue auto-closed & post completion summary:**
   ```bash
   gh issue view <N> --json state --jq .state
   gh issue edit <N> --remove-label "status:in-review"
   ```
4. **Unblock downstream issues in the DAG:**
   Query all open issues labeled `status:blocked` whose body mentions `Blocked by #<N>`:
   ```bash
   gh issue list --state open --label "status:blocked" --limit 100 \
     --json number,title,body
   ```
   For each candidate issue `#M`, inspect **all** `Blocked by #<DEP>` references in its `## Dependencies` section. If **every** `<DEP>` is now `CLOSED`, promote `#M` to `status:ready`:
   ```bash
   gh issue edit <M> --remove-label "status:blocked" --add-label "status:ready"
   gh issue comment <M> --body "All prerequisites are now closed (unblocked by #<N>). Promoted from \`status:blocked\` to \`status:ready\`."
   ```
