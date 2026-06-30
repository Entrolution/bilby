//! Error types for quadrature operations.

use core::fmt;

/// Errors that can occur during quadrature operations.
///
/// These represent truly unrecoverable conditions. For adaptive integration,
/// non-convergence is signalled via [`QuadratureResult::converged`](crate::QuadratureResult::converged)
/// rather than an error variant, since a best-effort estimate is always available.
#[derive(Debug, Clone, PartialEq)]
pub enum QuadratureError {
    /// The requested order `n` is zero.
    ZeroOrder,
    /// The integration interval is degenerate (a bound is non-finite, i.e. NaN
    /// or infinite).
    DegenerateInterval,
    /// An input parameter is invalid.
    InvalidInput(&'static str),
}

impl fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroOrder => write!(f, "quadrature order must be at least 1"),
            Self::DegenerateInterval => write!(f, "integration interval is degenerate"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

// `core::error::Error` (stable since Rust 1.81; MSRV is 1.93) is available
// without `std`, so the impl is unconditional and `no_std` users get it too.
impl core::error::Error for QuadratureError {}
