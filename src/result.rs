//! Result types for quadrature operations.

use num_traits::Float;

/// Outcome of a quadrature computation.
///
/// Always returned by adaptive integration methods. Contains the best estimate
/// even when convergence was not achieved. Check [`converged`](Self::converged)
/// to determine whether the requested tolerance was met; if it is `false`,
/// [`roundoff_limited`](Self::roundoff_limited) distinguishes hitting the
/// floating-point floor from exhausting the evaluation budget.
///
/// Marked `#[non_exhaustive]` so future fields can be added without a breaking
/// change; construct one only through the library's integration methods.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QuadratureResult<F> {
    /// The computed integral value.
    pub value: F,
    /// Estimated absolute error.
    pub error_estimate: F,
    /// Number of integrand evaluations.
    pub num_evals: usize,
    /// Whether the requested tolerance was achieved (i.e. `error_estimate` is
    /// within the requested absolute/relative tolerance).
    pub converged: bool,
    /// Whether integration stopped at the floating-point roundoff floor
    /// (analogous to QUADPACK's `ier = 2`): the requested tolerance was tighter
    /// than the accuracy achievable in `f64`, so further subdivision could not
    /// reduce `error_estimate`. When this is `true` the `value` is typically as
    /// accurate as floating-point allows. Only the one-dimensional Gauss-Kronrod
    /// adaptive integrator and its delegators (Cauchy principal value,
    /// one-dimensional cubature) ever set it; it is always `false` for the
    /// Monte-Carlo, tanh-sinh, oscillatory, and Genz-Malik cubature methods.
    /// Mutually exclusive with `converged`: if the tolerance was met this is
    /// `false`, so `converged` / `roundoff_limited` / neither are the three
    /// distinct stopping outcomes (the last being budget exhaustion).
    pub roundoff_limited: bool,
}

impl<F: Float> QuadratureResult<F> {
    /// Returns `true` if the integration converged within tolerance.
    #[inline]
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Returns `true` if integration stopped at the floating-point roundoff
    /// floor (the requested tolerance was unachievable in `f64`).
    #[inline]
    pub fn is_roundoff_limited(&self) -> bool {
        self.roundoff_limited
    }
}
