# Contributing to bilby

Thank you for your interest in contributing to bilby! This document provides guidelines and information for contributors.

## Code of Conduct

This project is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/bilby.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `cargo test`
6. Run lints: `cargo clippy && cargo fmt --check`
7. Commit your changes
8. Push to your fork and submit a pull request

## Development Setup

### Prerequisites

- Rust 1.83 or later (install via [rustup](https://rustup.rs/))
- Cargo (included with Rust)

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run a specific test
cargo test test_name
```

### Code Quality

Before submitting a PR, ensure:

```bash
# Format code
cargo fmt

# Run clippy
cargo clippy -- -D warnings

# Check documentation
cargo doc --no-deps
```

### Benchmarks

```bash
# Run benchmarks
cargo bench
```

### Security Audits

Run dependency audits before submitting PRs:

```bash
# Install audit tools (one-time)
cargo install cargo-audit cargo-deny

# Check for known vulnerabilities
cargo audit

# Check licenses and advisories
cargo deny check
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code is formatted with `cargo fmt`
- [ ] No clippy warnings
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated for user-facing changes

### PR Description

Please include:

- **What**: Brief description of the change
- **Why**: Motivation for the change
- **How**: High-level approach (if not obvious)
- **Testing**: How you tested the changes

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Adding New Features

1. **Discuss first**: Open an issue to discuss significant changes
2. **Backward compatibility**: Avoid breaking changes unless necessary
3. **Testing**: Add tests for new functionality — validate against known exact integrals and reference values
4. **Documentation**: Update rustdoc and README as needed
5. **Benchmarks**: Add benchmarks for performance-sensitive code

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for general questions

Thank you for contributing!
