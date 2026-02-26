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

### 3.1 Tensor product rules
- `QuadratureRule::tensor_product(rule_x, rule_y)` for 2D, generalise to N-D
- Simple but exponential cost (curse of dimensionality)
- Practical for d <= 4-5

### 3.2 Sparse grids (Smolyak)
- Break the curse of dimensionality for smooth integrands
- Smolyak construction from nested 1D rules (Clenshaw-Curtis preferred)
- `SparseGrid::new(dim, level)` -> nodes and weights
- Practical for moderate dimensions (d <= ~20 for smooth functions)

### 3.3 Adaptive cubature (Genz-Malik)
- h-adaptive: recursively subdivide the worst subregion
- Based on the Genz-Malik 7th-degree rule for hypercubes
- `adaptive_cubature(f, xmin, xmax, tol)` for d < 7ish
- Vectorised integrand interface for batch evaluation

### 3.4 Monte Carlo / quasi-Monte Carlo
- Plain MC with error estimate (1/sqrt(N) convergence)
- Quasi-MC: Sobol, Halton sequences (better convergence for smooth functions)
- Stratified sampling
- These dominate for d > ~10

**Milestone: v0.4.0** — Multi-dimensional integration.

## Phase 4: Specialty Methods

### 4.1 Oscillatory integrals
- Filon-type methods for integrals of f(x)*exp(i*omega*x) or f(x)*sin(omega*x)
- Levin collocation for more general oscillators
- Critical for Fourier transforms, wave problems, BEM

### 4.2 Double-exponential (tanh-sinh)
- Transforms endpoint singularities into rapidly decaying integrands
- Good for integrands with algebraic or logarithmic endpoint singularities
- Self-adaptive via level doubling

### 4.3 Cauchy principal value
- `pv_integrate(f, a, b, c, tol)` where c is the singularity
- Subtraction technique + adaptive integration of the remainder

### 4.4 Weighted integration
- Integrals of f(x)*w(x) where w(x) is a known weight (log, algebraic, etc.)
- Product integration rules

**Milestone: v0.5.0** — Specialty quadrature.

## Phase 5: Performance and Polish

### 5.1 Parallelism (`parallel` feature)
- Parallel adaptive subdivision (evaluate subintervals concurrently)
- Parallel node/weight generation for large n
- Vectorised integrand batching (evaluate f at multiple points per call)

### 5.2 Precomputed rule caching
- `LazyRule` / compile-time tables for common orders
- Avoid recomputing nodes/weights in hot loops

### 5.3 `no_std` core
- Core rule computations without alloc where feasible
- Fixed-size rules via const generics

### 5.4 Comprehensive benchmarks
- Against gauss-quad, quadrature, quad_gk
- Against SciPy/QUADPACK reference values
- Criterion benchmarks for node generation, integration, adaptive convergence

**Milestone: v1.0.0** — Production-ready.

## Non-Goals (at least initially)

- ODE/PDE solvers (different domain)
- Symbolic integration
- Arbitrary-precision arithmetic (stay in f32/f64 land)
- GPU acceleration (diminishing returns for quadrature vs. the integrand evaluation itself)

## Key Design Decisions to Make Early

1. **Trait bounds**: `F: Float` (num-traits) vs custom trait. Leaning toward num-traits `Float` for echidna compatibility.
2. **Node/weight storage**: `Vec<F>` vs `&[F]` vs const-generic arrays. Likely `Vec` for flexibility with a `SmallRule<F, N>` const-generic variant for known small sizes.
3. **Integrand signature**: `Fn(F) -> F` vs `Fn(&[F]) -> F` for multi-dimensional. Probably separate 1D and N-D APIs rather than a unified signature.
4. **Error model**: return `(value, error)` tuple vs a `QuadratureResult` struct. Struct is more extensible (can add num_evals, converged flag later).
5. **Rule representation**: separate types per family (GaussLegendre, GaussKronrod, etc.) vs enum. Separate types with a shared trait — more ergonomic and allows family-specific methods.

## References

- Bogaert (2014) — [Iteration-Free Computation of Gauss-Legendre Quadrature Nodes and Weights](https://www.cfm.brown.edu/faculty/gk/APMA2560/Handouts/GL_quad_Bogaert_2014.pdf)
- QUADPACK — Piessens, de Doncker-Kapenga, Uberhuber, Kahaner (1983)
- [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) — Julia reference implementation
- [cubature](https://github.com/stevengj/cubature) — Steven G. Johnson's adaptive cubature in C
- Genz & Malik (1980) — adaptive cubature rules for hypercubes
- [SciPy integrate](https://docs.scipy.org/doc/scipy/tutorial/integrate.html) — Python QUADPACK wrapper
- [sparse-grids.de](http://www.sparse-grids.de/) — Sparse grid reference tables and theory
