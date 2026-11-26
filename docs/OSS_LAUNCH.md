# OSS launch readiness

This is the checklist for making `donfuldauer/kggml` public (issue #24). The automated parts run on every pull
request. The steps that change repository settings or visibility can only be done by the owner. They are listed
under [Owner steps before the visibility flip](#owner-steps-before-the-visibility-flip) and in
[`MANUAL_VERIFICATION.md`](MANUAL_VERIFICATION.md).

## Full-history audit

[`scripts/ci/audit-history.sh`](../scripts/ci/audit-history.sh) scans every commit reachable from any branch, tag or
fetched pull-request head, not just the current tree. It fails if any commit ever contained:

| Check | What fails |
| --- | --- |
| Secrets | anything `gitleaks git --log-opts=--all` reports, using [`.gitleaks.toml`](../.gitleaks.toml) |
| Blob size | any blob over 1 MiB (`KGGML_HISTORY_MAX_BLOB_BYTES`) |
| Credential and local files | `.pem`, `.key`, `.p12`, `.pfx`, `.jks`, `.keystore`, `.gpg`, `.p8`, `.env` (except `.env.example`), `local.properties`, `.DS_Store`, SSH keys, `credentials.json`, `google-services.json`, `GoogleService-Info.plist`, even if a later commit deleted them |
| Binaries | a model, audio, image, archive or compiled file whose path is not in [`NOTICE`](../NOTICE) |
| Private keys | a `BEGIN … PRIVATE KEY` or PGP private key block in any file or commit message |
| Developer paths | a `/Users/<name>/`, `/home/<name>/`, root home directory (`/root`) or `C:\Users\<name>\` path in any file or commit message, except the placeholder names `someone`, `example`, `runner` and `username` (`KGGML_HISTORY_PATH_ALLOW`) |

It also reports, without failing, commits whose author or committer e-mail is not a GitHub or Anthropic no-reply
address. The address itself is never printed (see [Commit e-mail addresses](#commit-e-mail-addresses)).

It runs in the `secret-scan` job of [`repo-hygiene.yml`](../.github/workflows/repo-hygiene.yml) on every PR, over
every branch plus the PR's own head. Pull-request heads stay on GitHub read-only, so one bad fork PR must not fail
every later PR: a manual dispatch of `repo-hygiene.yml` audits every pull-request head too. It also runs in
`./scripts/ci-local.sh` (step `history`), and its tests are in
[`scripts/tests/test-audit-history.sh`](../scripts/tests/test-audit-history.sh). To run it by hand on a fresh full
clone:

```bash
git clone https://github.com/donfuldauer/kggml && cd kggml
git fetch origin '+refs/heads/*:refs/remotes/origin/*' '+refs/pull/*/head:refs/remotes/pull/*'
./scripts/ci/audit-history.sh
```

### Result at launch (issue #24)

Run on 2026-11-26 against `main` at `2c64e1e`, every branch and pull-request head, and this issue's branch (129
commits, 722 blobs at the time):

- gitleaks: no leaks, with and without the `.gitleaks.toml` allowlist.
- Largest blob: 49 KB (`kggml-llama/native/src/kggml_llama.cpp`). Nothing over 1 MiB.
- Binaries: the only binary ever committed is `gradle/wrapper/gradle-wrapper.jar`, recorded in `NOTICE`.
- No credential, key, keystore, `local.properties` or `.DS_Store` file in any commit, and no private-key block.
- Developer paths: no real machine path in any file or commit message. The only match is the placeholder
  `/home/someone/kggml` in the docs-audit test; the audit's own test builds its sample paths from pieces.
- Commit e-mails: 12 merge and early commits made by the owner use a personal (non-no-reply) address. This is an owner
  decision, not a finding (next section).
- Commit messages carry `Claude-Session:` attribution links. They open only for the signed-in owner and contain no
  credentials, so they stay.

**No `git filter-repo` remediation was needed.**

## Commit e-mail addresses

GitHub shows the author and committer e-mail of every commit in a public repository. Some commits made by the owner
(merges done locally or from the web UI before "Keep my email addresses private" was set) carry a personal address.
This is not a secret, but if the owner does not want it public, the options are:

1. **Keep history as is.** Turn on Settings → Emails → "Keep my email addresses private" and "Block command line
   pushes that expose my email" so that future commits use the no-reply address.
2. **Rewrite the address before the flip**, using the procedure below with a mailmap file instead of path filters:
   `git filter-repo --mailmap <file>`, where the file maps the personal address to
   `Don Fuldauer <13605136+donfuldauer@users.noreply.github.com>`. This changes every commit SHA after the first
   rewritten commit.

`./scripts/ci/audit-history.sh --strict-identity` fails while any such commit remains.

## Remediating a finding

If the audit ever reports a real finding, only the owner rewrites history, and only before the repository is public:

1. **Rotate first.** A leaked secret stays compromised even after it leaves history. Revoke or rotate it at the
   provider before anything else.
2. Merge or close every open pull request, and stop scheduled agent sessions.
3. In a fresh mirror clone, remove the finding. Install `git filter-repo` with `pip install git-filter-repo`.

   ```bash
   git clone --mirror https://github.com/donfuldauer/kggml kggml-rewrite.git && cd kggml-rewrite.git
   git filter-repo --invert-paths --path <file-to-drop>             # a file
   git filter-repo --replace-text <expressions-file>                # a string, e.g. 'literal==>REDACTED'
   git filter-repo --strip-blobs-bigger-than 1M                     # oversized blobs
   ```

4. Run `./scripts/ci/audit-history.sh --repo <rewritten clone>` until it passes, then push the result with
   `git push --force --mirror`. GitHub keeps pull-request refs (`refs/pull/*`) read-only. If the finding is in one,
   ask GitHub Support to purge the cached views and dangling objects.
5. Every collaborator re-clones. Pinned SHAs in issues and PR bodies no longer resolve.

## Launch checklist

| # | Item | Evidence | Status |
| --- | --- | --- | --- |
| 1 | MIT `LICENSE` | `scripts/ci/verify-licenses.sh` (every PR) | Verified (#21) |
| 2 | `NOTICE` records every upstream pin, the Gradle wrapper and every fixture (MIT/CC0) | `scripts/ci/verify-licenses.sh` (every PR) | Verified (#21) |
| 3 | Production `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates | `scripts/ci/verify-docs.sh` and `:docs-snippets` compile (every PR) | Verified (#22) |
| 4 | Maven Central staging: 7 coordinates signed, POM, `.module` and BOM metadata audited | `scripts/ci/stage-release.sh --dry-run` (every PR), [`RELEASING.md`](RELEASING.md) | Verified (#23); first real upload is an owner step |
| 5 | Every workflow job runs on a GitHub-hosted runner: the self-hosted routes (Mac, Windows, Linux) added back in #62 are removed | one PR (#63); `scripts/ci/verify-hosted-runners.sh` fails on every PR while public with any route left | **Open** (#63) |
| 6 | No runner registered to the repository and no runner variable (`MAC_RUNNER`, `WIN_RUNNER`, `LINUX_RUNNER`) | Settings → Actions → Runners | Owner step (below, #63) |
| 7 | Full-history gitleaks, blob, binary, credential and path audit clean | `scripts/ci/audit-history.sh` (every PR), result above | Verified (#24) |
| 8 | Agent and harness instruction mirrors in sync | `scripts/verify-agent-parity.sh` (every PR) | Verified |
| 9 | GitHub secret scanning and push protection on | Settings → Code security | Owner step (below) |
| 10 | Full CI matrix (Ubuntu, macOS, Windows) green on hosted runners from `main` | `ci.yml` manual dispatch | Owner step (Actions minutes) |
| 11 | `phase-reviewer` OSS launch gate | issue #24 PR | `APPROVED` (see the PR) |

## Owner steps before the visibility flip

1. **Runners (#63).** Unregister all three self-hosted runners: the Mac (`local-mac`, on the Mac
   `gh-runner remove donfuldauer/kggml`), the Windows PC (`local-win`: delete its logon task, then `config.cmd remove`
   in the runner folder) and the Linux VM (`local-linux`: stop its service, then `./config.sh remove`); removal tokens
   come from Settings → Actions → Runners. Delete the variables `MAC_RUNNER`, `WIN_RUNNER` and `LINUX_RUNNER` under
   Settings → Secrets and variables → Actions → Variables, and merge the #63 PR that drops the self-hosted routes from
   the workflows. With the GitHub CLI: `gh api repos/donfuldauer/kggml/actions/runners --jq .total_count` prints `0`,
   and `gh variable list` shows no runner variable.
2. **Secret scanning.** Settings → Code security: turn on "Secret Protection" (secret scanning) and "Push
   protection". Both are free on public repositories and take effect when the repository becomes public, if not
   before. After the flip, Security → Secret scanning should list no alerts.
3. **Commit e-mails.** Decide on [Commit e-mail addresses](#commit-e-mail-addresses).
4. **Audit once more** on a fresh full clone, as above, right before the flip.
5. **Flip.** Settings → General → Danger Zone → Change visibility → Public.
6. **After the flip.** Run `./scripts/ci/verify-docs.sh --online --only links` (from #22) and dispatch the full
   `ci.yml` matrix from `main` (from #23).
