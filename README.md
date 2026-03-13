# bilby

[![CI](https://github.com/Entrolution/bilby/actions/workflows/ci.yml/badge.svg)](https://github.com/Entrolution/bilby/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/bilby.svg)](https://crates.io/crates/bilby)
[![Docs.rs](https://docs.rs/bilby/badge.svg)](https://docs.rs/bilby)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![MSRV](https://img.shields.io/badge/MSRV-1.93-blue.svg)](https://www.rust-lang.org)

A high-performance numerical quadrature (integration) library for Rust.

- **Comprehensive** -- Gauss-Legendre, Gauss-Kronrod, Jacobi, Hermite, Laguerre, Chebyshev, Radau, Lobatto, Clenshaw-Curtis, tanh-sinh, oscillatory, Cauchy PV
- **Adaptive integration** -- QUADPACK-style error-driven refinement with configurable GK pairs
- **Multi-dimensional** -- tensor products, Smolyak sparse grids, Genz-Malik adaptive cubature, quasi-Monte Carlo (Sobol, Halton)
- **Generic over `F: Float`** -- works with `f32`, `f64`, and AD types (e.g., [echidna](https://github.com/Entrolution/echidna))
- **`no_std` compatible** -- works without the standard library (with `alloc`)
- **Optional parallelism** -- `parallel` feature for rayon-based `_par` methods
- **Precomputed rules** -- generate nodes/weights once, integrate many times with zero allocation

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
bilby = "0.2"
```

```rust
use bilby::GaussLegendre;

// Create a 10-point Gauss-Legendre rule
let gl = GaussLegendre::new(10).unwrap();

// Integrate x^2 over [0, 1] (exact result = 1/3)
let result = gl.rule().integrate(0.0, 1.0, |x: f64| x * x);
assert!((result - 1.0 / 3.0).abs() < 1e-14);
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Enables `std::error::Error` impl and `cache` module |
| `parallel` | No | Enables rayon-based `_par` methods (implies `std`) |

### `no_std` support

bilby works in `no_std` environments (with `alloc`). Disable default features:

```toml
[dependencies]
bilby = { version = "0.2", default-features = false }
```

### Parallelism

Enable the `parallel` feature for parallel variants of integration methods:

```toml
[dependencies]
bilby = { version = "0.2", features = ["parallel"] }
```

This provides `integrate_composite_par`, `integrate_par`, `integrate_box_par`,
`MonteCarloIntegrator::integrate_par`, and `GaussLegendre::new_par`.

## Development

```bash
cargo test                         # Run tests (default features)
cargo test --all-features          # Run tests including parallel
cargo test --no-default-features   # Run tests in no_std mode
cargo bench                        # Run benchmarks
cargo clippy --all-features        # Lint
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
