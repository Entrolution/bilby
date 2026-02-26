# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

Initial release of bilby.

### Added

- **Gauss-Legendre** quadrature with Bogaert (2014) asymptotic expansion for large n and Newton iteration for small n
- **Gauss-Kronrod** embedded pairs (G7K15, G10K21, G15K31, G20K41, G25K51, G30K61) for error estimation
- **Classical families** via Golub-Welsch eigenvalue solver:
  - Gauss-Jacobi (arbitrary alpha, beta)
  - Gauss-Hermite (physicists' convention)
  - Gauss-Laguerre (generalised, arbitrary alpha)
  - Gauss-Chebyshev Types I and II (closed-form)
  - Gauss-Lobatto (both endpoints)
  - Gauss-Radau (left or right endpoint)
  - Clenshaw-Curtis (nested, Chebyshev extrema)
- **Adaptive integration** -- QUADPACK-style global subdivision with configurable GK pair, break points, absolute/relative tolerance
- **Domain transforms** -- `integrate_infinite`, `integrate_semi_infinite_lower`, `integrate_semi_infinite_upper` for unbounded domains
- **Tanh-sinh** (double-exponential) quadrature for endpoint singularities
- **Cauchy principal value** integration via the subtraction technique
- **Oscillatory integration** via Filon-Clenshaw-Curtis with adaptive fallback for small omega
- **Weighted integration** -- unified API for product integration with Jacobi, Laguerre, Hermite, Chebyshev, and log weight functions
- **Multi-dimensional cubature**:
  - Tensor product rules (isotropic and anisotropic)
  - Smolyak sparse grids with Clenshaw-Curtis basis
  - Genz-Malik adaptive cubature (degree-7/degree-5 embedded rule)
  - Monte Carlo (plain pseudo-random with Welford variance estimation)
  - Quasi-Monte Carlo via Sobol and Halton low-discrepancy sequences
- **Generic over `F: Float`** via num-traits for core rule types
- **`no_std` compatible** -- works with `alloc` only (disable default `std` feature)
- **`parallel` feature** -- rayon-based parallel variants:
  - `GaussLegendre::new_par` for parallel node generation
  - `integrate_composite_par` for parallel composite rules
  - `integrate_par` / `integrate_box_par` for cubature rules
  - `MonteCarloIntegrator::integrate_par` for parallel MC/QMC
- **Precomputed rule cache** (`cache` module, requires `std`) with `GL5`, `GL10`, `GL15`, `GL20`, `GL50`, `GL100` lazy singletons
- **Benchmarks** via criterion for node generation, 1D integration, and cubature
