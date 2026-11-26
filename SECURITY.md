# Security Policy

## Supported versions

Security fixes are made on `main` and released in the next version. Only the latest released version of the Kggml
artifacts receives fixes.

## Reporting a vulnerability

Please **do not** report security problems in public issues, discussions or pull requests.

Report them privately through GitHub's
[private vulnerability reporting](https://github.com/donfuldauer/kggml/security/advisories/new) for this repository.
Include:

- the affected module(s) (`kggml-whisper`, `kggml-llama`, ...) and platform(s) (Android, iOS, JVM Desktop);
- the Kggml version or commit, and the upstream library involved if you know it;
- steps or a minimal input that reproduces the problem (for example a crafted model header or audio buffer);
- the impact you expect, and a suggested fix if you have one.

You can expect an acknowledgement within 7 days and an assessment within 30 days. Once a fix is released, the
advisory is published and credits the reporter unless you prefer to stay anonymous.

## Scope

In scope:

- memory-safety bugs at the JNI and Kotlin/Native boundaries (out-of-bounds access, use after free, leaks of native
  buffers) reachable through the public API;
- crashes or code execution caused by untrusted inputs that Kggml parses itself: model headers read by
  `kggml-model-manager`, KV-cache sessions passed to `LlamaEngine.restoreSession`, audio and image buffers;
- problems in the download code (`ModelDownloader`): skipped SHA-256 checks, path traversal, unsafe resumes;
- secrets or credentials committed to the repository.

Bugs in the upstream projects (ggml, whisper.cpp, llama.cpp, stable-diffusion.cpp) are usually best reported to
them as well. Tell us too, so the pinned commit in [NOTICE](NOTICE) can be updated.

Model files are code-like inputs: only load models and sessions from sources you trust, and set
`expectedSha256` when you download them.

## How the project protects itself

- `gitleaks` scans every pull request for secrets, and `.gitignore` excludes signing keys, credentials and
  environment files.
- Every C/C++ entry point catches exceptions and returns error codes, so no C++ exception crosses into Kotlin.
- Upstream libraries are pinned to exact commits, recorded in [NOTICE](NOTICE) and checked by CI.
