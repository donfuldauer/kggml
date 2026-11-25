## Summary

<!-- Describe what changed and why. -->

Closes #<ISSUE_NUMBER>

## Changes

-

## Verification

- [ ] `./scripts/ci-local.sh` passes (paste its summary block below)
- [ ] `./scripts/verify-agent-parity.sh` passes (if agent/instructions touched)
- [ ] Pure Kotlin unit tests (`commonTest`) pass
- [ ] Native C++ tests (`ctest`) pass (if native code touched)
- [ ] Platform integration / UI / binary symbol audit checks pass (as applicable)
- [ ] `NOTICE` updated if upstream C/C++ dependencies or test fixtures changed
- [ ] Manual / physical-device checks (if any) documented and listed in `docs/MANUAL_VERIFICATION.md` — deferred to final acceptance, not a merge gate
- [ ] `ci:macos` label added (phase-final PRs, or changes to macOS/iOS-specific native code)

```text
<ci-local.sh summary>
```

## Phase Gate

- [ ] Not a phase-final issue (routine mid-phase PR)
- [ ] Phase-final issue — `phase-reviewer` audit completed and `APPROVED`
