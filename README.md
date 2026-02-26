# bilby

[![CI](https://github.com/Entrolution/bilby/actions/workflows/ci.yml/badge.svg)](https://github.com/Entrolution/bilby/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/bilby.svg)](https://crates.io/crates/bilby)
[![Docs.rs](https://docs.rs/bilby/badge.svg)](https://docs.rs/bilby)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.83-blue.svg)](https://www.rust-lang.org)

A high-performance numerical quadrature (integration) library for Rust.

- **Gauss-Legendre** -- O(1) per-node generation via the Bogaert (2014) algorithm
- **Gauss-Kronrod** -- embedded pairs (G7-K15 through G25-K51) for error estimation
- **Generic over `F: Float`** -- works with `f32`, `f64`, and AD types (e.g., [echidna](https://github.com/Entrolution/echidna))
- **Precomputed rules** -- generate nodes/weights once, integrate many times with zero allocation

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
bilby = "0.1"
```

```rust,ignore
use bilby::GaussLegendre;

// Create a 10-point Gauss-Legendre rule
let gl = GaussLegendre::new(10);

// Integrate x^2 over [0, 1]
let result = gl.integrate(0.0, 1.0, |x| x * x);
assert!((result - 1.0 / 3.0).abs() < 1e-14);
```

## Roadmap

bilby is being developed in phases. See [ROADMAP.md](ROADMAP.md) for details.

| Phase | Version | Content |
|-------|---------|---------|
| 0 | v0.1.0 | Gauss-Legendre (Bogaert), Gauss-Kronrod pairs, fixed-order integration |
| 1 | v0.2.0 | Adaptive integration (QUADPACK-style), infinite domains, error control |
| 2 | v0.3.0 | Classical rule families (Jacobi, Laguerre, Hermite, Chebyshev, Radau, Lobatto, Clenshaw-Curtis) |
| 3 | v0.4.0 | Multi-dimensional (tensor product, sparse grids, adaptive cubature, quasi-Monte Carlo) |
| 4 | v0.5.0 | Specialty methods (oscillatory, tanh-sinh, Cauchy principal value) |
| 5 | v1.0.0 | Parallelism, caching, `no_std`, benchmarks |

## Development

```bash
cargo test          # Run tests
cargo bench         # Run benchmarks
cargo clippy        # Lint
cargo fmt           # Format
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
