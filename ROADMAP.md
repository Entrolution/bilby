# bilby — Numerical Quadrature for Rust

A high-performance, comprehensive numerical integration library for Rust.

**Design principles:**
- Generic over `F: Float` (composable with echidna AD types out of the box)
- No heap allocation on hot paths where possible
- Precomputable rules: separate node/weight generation from integration
- Consistent API: all rules return `(nodes, weights)` or `Result<T, QuadratureError>`
- Feature-gated heavy dependencies (rayon, etc.)

## Phase 0: Foundation

Core types, trait design, and the first two rule families. This phase defines the API that everything else builds on.

### 0.1 Project scaffolding ✅
- `cargo init --lib`, dual MIT/Apache-2.0, MSRV 1.93
- CI (clippy, fmt, test, MSRV check, coverage, security audit, publish workflow)
- `QuadratureRule<F>` struct with `nodes: Vec<F>`, `weights: Vec<F>`
- `QuadratureError` enum (ZeroOrder, DegenerateInterval, NotConverged)

### 0.2 Gauss-Legendre (Bogaert algorithm) ✅
- Newton iteration on Legendre recurrence for n <= 100 (Tricomi initial guess)
- Bogaert (2014) asymptotic expansion for n > 100 (O(1) per node via Bessel zeros + Chebyshev interpolants)
- `GaussLegendre::new(n) -> Result<GaussLegendre, QuadratureError>`
- 16-digit accuracy verified against exact polynomial integration up to degree 2n-1

### 0.3 Gauss-Kronrod (embedded pairs) ✅
- G7-K15, G10-K21, G15-K31, G20-K41, G25-K51 pairs
- Canonical QUADPACK coefficients (from netlib Fortran source)
- Returns `(estimate, error_estimate)` with QUADPACK error heuristic
- `GaussKronrod::new(pair)` with static coefficient slices (no allocation)

### 0.4 Basic integration API ✅
- `QuadratureRule::integrate(a, b, f)` — affine transform from [-1, 1] to [a, b]
- `QuadratureRule::integrate_composite(a, b, n_panels, f)` — composite rule
- `GaussKronrod::integrate(a, b, f)` — returns (estimate, error_estimate)
- 22 unit tests + 4 doc tests

**Milestone: v0.1.0** — Gauss-Legendre + Gauss-Kronrod, fixed-order integration.

## Phase 1: Adaptive Integration

The workhorse feature. Most real integration problems need error-driven refinement.

### 1.1 Global adaptive subdivision (QUADPACK-style) ✅
- `BinaryHeap`-based priority queue of subintervals ordered by error estimate
- Bisects the interval with largest error, re-evaluates both halves via GK pairs
- Terminates when `error <= max(abs_tol, rel_tol * |estimate|)` or max evaluations reached
- `AdaptiveIntegrator` builder with configurable GK pair, tolerances, and eval budget
- `adaptive_integrate(f, a, b, tol)` convenience function

### 1.2 Singularity handling ✅
- User-specified break points via `adaptive_integrate_with_breaks(f, a, b, breaks, tol)`
- Break points seed the priority queue as separate sub-intervals for global adaptive refinement
- Validation: breaks must be within (a, b), deduplicated and sorted automatically
- Endpoint singularity detection deferred to future refinement

### 1.3 Semi-infinite and infinite intervals ✅
- Domain transforms: [a, ∞) via x = a + t/(1-t), (-∞, b] via x = b - t/(1-t), (-∞, ∞) via x = t/(1-t²)
- `integrate_semi_infinite_upper(f, a, tol)`, `integrate_semi_infinite_lower(f, b, tol)`, `integrate_infinite(f, tol)`
- Delegates to adaptive quadrature on the transformed finite domain
- Gauss-Laguerre/Hermite as dedicated rule families deferred to Phase 2

### 1.4 Error reporting ✅
- `QuadratureResult<F>` struct: value, error_estimate, num_evals, converged flag
- Non-convergence signalled via `converged: false` (always returns best estimate)
- `QuadratureError` simplified: `NotConverged` removed, `InvalidInput` added
- 43 unit tests + 13 doc tests

**Milestone: v0.2.0** — Adaptive integration with error control, infinite domains.

## Phase 2: Classical Rule Families

Complete the Gaussian quadrature family.

### 2.1 Gauss-Jacobi ✅
- Weight function (1-x)^alpha * (1+x)^beta on [-1, 1]
- Generalises Legendre (alpha=beta=0), Chebyshev (alpha=beta=-0.5), Gegenbauer
- Golub-Welsch algorithm: eigenvalues of symmetric tridiagonal Jacobi matrix
- Handles all valid parameters including the singular Chebyshev limit (alpha+beta=-1)

### 2.2 Gauss-Laguerre (generalised) ✅
- Weight function x^alpha * e^(-x) on [0, inf)
- Golub-Welsch algorithm via Laguerre three-term recurrence coefficients

### 2.3 Gauss-Hermite ✅
- Weight function e^(-x^2) on (-inf, inf)
- Golub-Welsch algorithm via physicists' Hermite recurrence coefficients

### 2.4 Gauss-Chebyshev (types I and II) ✅
- Closed-form nodes and weights (no iteration needed)
- Useful as comparison/baseline and for Chebyshev interpolation

### 2.5 Gauss-Radau and Gauss-Lobatto ✅
- Gauss-Radau: Golub-Welsch with Radau modification of Legendre Jacobi matrix
- Gauss-Lobatto: Newton on P'_{n-1}(x) using Legendre ODE second derivative
- Left/right variants for Radau, both endpoints for Lobatto

### 2.6 Clenshaw-Curtis ✅
- Nodes at Chebyshev extrema cos(kπ/(n-1)), weights via explicit DCT formula
- Nested: degree doubling reuses previous evaluations
- Competitive with Gauss for smooth functions, better for adaptive refinement

### 2.7 Golub-Welsch eigenvalue solver ✅
- Shared `pub(crate)` module: symmetric tridiagonal QL algorithm with implicit shifts
- Radau modification via continued fraction on characteristic polynomial
- Used by Gauss-Jacobi, Gauss-Hermite, Gauss-Laguerre, and Gauss-Radau
- 98 unit tests + 21 doc tests

**Milestone: v0.3.0** — Full classical quadrature family.

## Phase 3: Multi-Dimensional Integration

### 3.1 Tensor product rules ✅
- `TensorProductRule::new(&[&QuadratureRule<f64>])` for mixed-order N-D rules
- `TensorProductRule::isotropic(rule, dim)` for same rule in all dimensions
- `CubatureRule` with flat node storage, `integrate()` and `integrate_box()` methods
- Simple but exponential cost (curse of dimensionality); practical for d <= 4-5

### 3.2 Sparse grids (Smolyak) ✅
- Smolyak combination technique from nested Clenshaw-Curtis rules
- `SparseGrid::clenshaw_curtis(dim, level)` with quantised point merging
- Practical for moderate dimensions (d <= ~20 for smooth functions)
- Verified: correct point counts, weight sums, polynomial exactness

### 3.3 Adaptive cubature (Genz-Malik) ✅
- h-adaptive: `BinaryHeap`-based subdivision of worst subregion
- Genz-Malik degree-7/5 embedded rule pair for error estimation
- Fourth-difference criterion for split axis selection
- `adaptive_cubature(f, lower, upper, tol)` convenience function
- d=1 delegates to 1D adaptive integrator

### 3.4 Monte Carlo / quasi-Monte Carlo ✅
- Plain pseudo-random MC with Welford's online variance (Xoshiro256++ PRNG)
- Quasi-MC: Sobol sequences (gray-code enumeration, 40 dimensions, Joe-Kuo direction numbers)
- Quasi-MC: Halton sequences (radical-inverse function, 100 primes)
- `MonteCarloIntegrator` builder with `Plain`/`Sobol`/`Halton` methods
- Heuristic error estimate for QMC via N/2 vs N comparison
- 136 unit tests + 28 doc tests

**Milestone: v0.4.0** — Multi-dimensional integration.

## Phase 4: Specialty Methods

### 4.1 Oscillatory integrals ✅
- Filon-Clenshaw-Curtis method for ∫f(x)sin(ωx)dx and ∫f(x)cos(ωx)dx
- Chebyshev expansion of f at Clenshaw-Curtis nodes
- Modified Chebyshev moments computed via Gauss-Legendre quadrature
- Phase-shifted combination for sin/cos kernels; small-ω fallback to adaptive
- `OscillatoryIntegrator` builder + `integrate_oscillatory_sin/cos` convenience functions

### 4.2 Double-exponential (tanh-sinh) ✅
- Transform x = tanh(π/2·sinh(t)) converts endpoint singularities to rapid decay
- Self-adaptive via level doubling: each level halves step size, reuses all prior points
- Handles algebraic (x^{-0.75}) and logarithmic (ln x) endpoint singularities
- `TanhSinh` builder + `tanh_sinh_integrate` convenience function

### 4.3 Cauchy principal value ✅
- PV ∫f(x)/(x-c)dx = ∫[f(x)-f(c)]/(x-c)dx + f(c)·ln((b-c)/(c-a))
- Subtraction technique with adaptive integration of the regularised remainder
- Automatic breakpoint at the pole via `adaptive_integrate_with_breaks`
- `CauchyPV` builder + `pv_integrate` convenience function

### 4.4 Weighted integration ✅
- Unified API: `∫f(x)·w(x)dx` dispatching to appropriate Gaussian rule
- Jacobi, Laguerre, Hermite, Chebyshev I/II weight families
- LogWeight w(x)=-ln(x) via Gauss-Laguerre α=1 substitution (x=e^{-t})
- `WeightedIntegrator` builder + `weighted_integrate` convenience function
- `integrate_over(a, b, f)` for affine-mapped finite domains
- 172 unit tests + 35 doc tests

**Milestone: v0.5.0** — Specialty quadrature.

## Phase 5: Performance and Polish ✅

### 5.1 Comprehensive benchmarks ✅
- Criterion benchmarks: node generation, 1D integration, cubature
- GL Newton/Bogaert, Golub-Welsch, Clenshaw-Curtis construction costs
- Fixed-order, composite, GK, adaptive, tanh-sinh, oscillatory throughput
- Tensor product, sparse grid, adaptive cubature, Monte Carlo (plain/Sobol/Halton)

### 5.2 `no_std` support ✅
- `#![cfg_attr(not(feature = "std"), no_std)]` with `alloc` for heap types
- `default = ["std"]` feature gate; `std::error::Error` impl gated behind `std`
- `num-traits` with `libm` feature for math functions in `no_std` mode
- `BTreeMap` replaces `HashMap` in sparse grid for `no_std` compatibility

### 5.3 Parallelism (`parallel` feature) ✅
- `QuadratureRule::integrate_composite_par` — parallel panel evaluation
- `CubatureRule::integrate_par` / `integrate_box_par` — parallel point evaluation
- `MonteCarloIntegrator::integrate_par` — parallel MC/QMC (Sobol/Halton deterministic)
- `GaussLegendre::new_par` — parallel node generation (Newton + Bogaert)
- `_par` suffix convention: explicit opt-in avoids `Sync` bound on existing API

### 5.4 Caching and polish ✅
- `cache` module: `LazyLock`-based precomputed GL rules (GL5–GL100)
- `#[inline]` on all trivial getters and hot-path integration functions
- 179 unit tests + 38 doc tests

**Milestone: v0.1.0** — Performance and polish.

## Non-Goals (at least initially)

- ODE/PDE solvers (different domain)
- Symbolic integration
- Arbitrary-precision arithmetic (stay in f32/f64 land)
- GPU acceleration (diminishing returns for quadrature vs. the integrand evaluation itself)

## Key Design Decisions

1. **Trait bounds**: `F: Float` via `num_traits::Float` — composable with echidna AD types
2. **Node/weight storage**: `Vec<F>` for flexibility; fixed-size const-generic variants not yet needed
3. **Integrand signature**: `Fn(F) -> F` for 1D, `Fn(&[F]) -> F` for N-D — separate APIs
4. **Error model**: `QuadratureResult<F>` struct with value, error estimate, num evals, convergence flag
5. **Rule representation**: separate types per family (`GaussLegendre`, `GaussKronrod`, etc.) with shared `QuadratureRule<F>` struct

## References

- Bogaert (2014) — [Iteration-Free Computation of Gauss-Legendre Quadrature Nodes and Weights](https://www.cfm.brown.edu/faculty/gk/APMA2560/Handouts/GL_quad_Bogaert_2014.pdf)
- QUADPACK — Piessens, de Doncker-Kapenga, Uberhuber, Kahaner (1983)
- [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) — Julia reference implementation
- [cubature](https://github.com/stevengj/cubature) — Steven G. Johnson's adaptive cubature in C
- Genz & Malik (1980) — adaptive cubature rules for hypercubes
- [SciPy integrate](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) — Python QUADPACK wrapper
- [sparse-grids.de](http://www.sparse-grids.de/) — Sparse grid reference tables and theory
