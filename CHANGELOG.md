# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Added

- Adaptive integration now detects the floating-point roundoff floor (analogous
  to QUADPACK's `ier = 2`): when the requested tolerance is tighter than the
  accuracy achievable in `f64`, it stops subdividing instead of spinning to
  `max_evals`. The per-panel Gauss-Kronrod error estimate gains the QUADPACK
  `max(50·εmach·∫|f|, error)` floor so a saturated panel reports an honest error.

### Changed

- **Breaking:** `QuadratureResult` gains a `roundoff_limited: bool` field (with an
  `is_roundoff_limited()` accessor) distinguishing "stopped at the floating-point
  floor" from "exhausted the evaluation budget"; `converged` now means strictly
  "requested tolerance met". The struct is now `#[non_exhaustive]`, so subsequent
  field additions will not be breaking.

### Fixed

- This release also rolls up a number of correctness and robustness fixes: the
  corrupted Sobol direction-number table, the `resasc`-based Gauss-Kronrod error
  estimate, oscillatory frequency bounding, tanh-sinh handling of reversed
  limits, the Jacobi recurrence coefficients, adaptive and cubature resource
  bounds, transform-endpoint guards, stricter input validation, and
  Golub-Welsch eigensolver guards.

## [0.2.0] - 2026-03-13

### Changed

- **Breaking:** `golub_welsch()` now returns `Result`, propagating QL non-convergence as `QuadratureError::InvalidInput` instead of silently returning inaccurate nodes/weights. All internal `compute_*` functions in `gauss_jacobi`, `gauss_laguerre`, `gauss_hermite`, and `gauss_radau` propagate accordingly. Public constructors already returned `Result`, so most callers are unaffected.
- **Breaking:** `GaussLobatto::new(0)` now returns `QuadratureError::ZeroOrder` (previously `InvalidInput`). `GaussLobatto::new(1)` still returns `InvalidInput`.
- Input validation for `tanh_sinh`, `oscillatory`, `cauchy_pv`, and `cubature::adaptive` now rejects `±Inf` bounds (not just `NaN`). Error variant is unchanged (`DegenerateInterval`).
- `CubatureRule::new` assertion promoted from `debug_assert_eq!` to `assert_eq!`.
- Tanh-sinh non-convergence error estimate now uses the difference between the last two level estimates instead of a fabricated `tol * 10` value.
- QUADPACK error estimation heuristic in `gauss_kronrod` documented as an intentional simplification of the full formula.

### Fixed

- `ln_gamma` reflection formula: `.sin().ln()` → `.sin().abs().ln()` prevents `NaN` for negative arguments where `sin(π·x)` is negative (affects Gauss-Jacobi with certain α, β near negative integers).
- Newton iteration in Gauss-Lobatto interior node computation now clamps iterates to `(-1+ε, 1-ε)`, preventing division by zero in the `P''` formula when an iterate lands on ±1.
- `partial_cmp().unwrap()` in `golub_welsch` and `gauss_lobatto` sort replaced with `.unwrap_or(Ordering::Equal)` to avoid panics on NaN nodes.
- Adaptive cubature `global_error` subtraction clamped to `max(0.0)` to prevent negative error estimates from floating-point cancellation.

### Added

- Dimension cap (d ≤ 30) for adaptive cubature — returns `InvalidInput` for higher dimensions where Genz-Malik's `2^d` vertex evaluations would be impractical.
- New tests: infinity rejection (tanh-sinh, oscillatory, Cauchy PV, cubature), dimension cap, Lobatto error variant splitting, tanh-sinh non-fabricated error, Jacobi negative α/β exercising the reflection path.

## [0.1.0] - 2026-03-12

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
