# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

Only the latest minor release receives security updates.

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue.
2. Email **security@entrolution.com** with details of the vulnerability.
3. Include steps to reproduce, if possible.

We aim to acknowledge reports within 48 hours and provide a fix or mitigation within 7 days for critical issues.

## Security Practices

- All floating-point operations use standard Rust primitives.
- No `unsafe` code.
- NaN/Inf propagation follows IEEE 754 semantics.
