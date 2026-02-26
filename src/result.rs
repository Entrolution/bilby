//! Result types for quadrature operations.

use num_traits::Float;

/// Outcome of a quadrature computation.
///
/// Always returned by adaptive integration methods. Contains the best estimate
/// even when convergence was not achieved — check [`converged`](Self::converged)
/// to determine whether the requested tolerance was met.
#[derive(Debug, Clone)]
pub struct QuadratureResult<F> {
    /// The computed integral value.
    pub value: F,
    /// Estimated absolute error.
    pub error_estimate: F,
    /// Number of integrand evaluations.
    pub num_evals: usize,
    /// Whether the requested tolerance was achieved.
    pub converged: bool,
}

impl<F: Float> QuadratureResult<F> {
    /// Returns `true` if the integration converged within tolerance.
    #[inline]
    pub fn is_converged(&self) -> bool {
        self.converged
    }
}
