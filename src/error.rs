//! Error types for quadrature operations.

use core::fmt;

/// Errors that can occur during quadrature operations.
#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureError {
    /// The requested order `n` is zero.
    ZeroOrder,
    /// The integration interval is degenerate (a == b or NaN).
    DegenerateInterval,
    /// Adaptive integration did not converge within the allowed evaluations.
    NotConverged {
        /// Best estimate so far.
        estimate: f64,
        /// Estimated error.
        error_estimate: f64,
        /// Number of function evaluations used.
        num_evals: usize,
    },
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroOrder => write!(f, "quadrature order must be at least 1"),
            Self::DegenerateInterval => write!(f, "integration interval is degenerate"),
            Self::NotConverged {
                estimate,
                error_estimate,
                num_evals,
            } => write!(
                f,
                "did not converge after {num_evals} evaluations \
                 (estimate = {estimate}, error ~ {error_estimate})"
            ),
        }
    }
}

impl std::error::Error for QuadratureError {}
