# Security Policy

## Reporting a Vulnerability

Please **do not** report security vulnerabilities (including memory-safety issues across JNI/Kotlin/Native boundaries, native crashes exploitable via untrusted model/audio inputs, or credential exposure) in public GitHub issues.

Instead, report them privately via GitHub Security Advisories for `donfuldauer/kggml` (or contact the maintainer directly) with:

- Affected module(s) and platform target(s) (`Android`, `iOS`, `JVM Desktop`).
- Minimal reproduction steps or test fixture details.
- Potential impact and suggested mitigation, if known.

## Security & Secret Hygiene Practices

- Automated `gitleaks` scanning runs on pushes and pull requests.
- Signing keys, Maven Central credentials, and environment files are strictly excluded via `.gitignore`.
- All C++ FFI entry points must catch exceptions across `extern "C"` boundaries and validate model/buffer bounds before native allocation.
