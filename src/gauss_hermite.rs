//! Gauss-Hermite quadrature rule.
//!
//! Weight function `w(x) = e^(-x²)` on (-∞, ∞).
//!
//! Nodes and weights are computed via the Golub-Welsch algorithm using
//! the physicists' Hermite polynomial three-term recurrence.

use crate::error::QuadratureError;
use crate::golub_welsch::golub_welsch;
use crate::rule::QuadratureRule;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// A Gauss-Hermite quadrature rule.
///
/// Integrates `f(x) * e^(-x²)` over (-∞, ∞) using n points.
///
/// # Example
///
/// ```
/// use bilby::GaussHermite;
///
/// // Integral of e^(-x^2) over (-inf, inf) = sqrt(pi)
/// let gh = GaussHermite::new(10).unwrap();
/// let result: f64 = gh.nodes().iter().zip(gh.weights()).map(|(_, &w)| w).sum();
/// assert!((result - core::f64::consts::PI.sqrt()).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct GaussHermite {
    rule: QuadratureRule<f64>,
}

impl GaussHermite {
    /// Create a new n-point Gauss-Hermite rule.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::ZeroOrder`] if `n` is zero.
    pub fn new(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }

        let (nodes, weights) = compute_hermite(n);
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
    }
}

impl_rule_accessors!(GaussHermite, nodes_doc: "Returns the nodes on (-∞, ∞).");

/// Compute n Gauss-Hermite nodes and weights via Golub-Welsch.
///
/// Monic physicists' Hermite recurrence:
///   x `h̃_k` = `h̃_{k+1}` + 0·`h̃_k` + (k/2)·`h̃_{k-1}`
///
/// Jacobi matrix: diagonal = 0, off-diagonal² = (k+1)/2 for k=0..n-2.
/// μ₀ = √π.
#[allow(clippy::cast_precision_loss)] // n is a quadrature order, always small enough for exact f64
fn compute_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let diag = vec![0.0; n];
    let off_diag_sq: Vec<f64> = (1..n).map(|k| k as f64 / 2.0).collect();
    let mu0 = core::f64::consts::PI.sqrt();

    golub_welsch(&diag, &off_diag_sq, mu0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn zero_order() {
        assert!(GaussHermite::new(0).is_err());
    }

    /// Weight sum = integral of e^(-x^2) over (-inf, inf) = sqrt(pi).
    #[test]
    fn weight_sum() {
        let gh = GaussHermite::new(20).unwrap();
        let sum: f64 = gh.weights().iter().sum();
        assert!((sum - PI.sqrt()).abs() < 1e-12, "sum={sum}");
    }

    /// Nodes should be symmetric about 0.
    #[test]
    fn node_symmetry() {
        let gh = GaussHermite::new(21).unwrap();
        let n = gh.order();
        for i in 0..n / 2 {
            assert!(
                (gh.nodes()[i] + gh.nodes()[n - 1 - i]).abs() < 1e-12,
                "i={i}: {} vs {}",
                gh.nodes()[i],
                gh.nodes()[n - 1 - i]
            );
        }
        // Middle node should be ~0 for odd n
        if n % 2 == 1 {
            assert!(gh.nodes()[n / 2].abs() < 1e-14);
        }
    }

    /// Weight symmetry.
    #[test]
    fn weight_symmetry() {
        let gh = GaussHermite::new(20).unwrap();
        let n = gh.order();
        for i in 0..n / 2 {
            assert!(
                (gh.weights()[i] - gh.weights()[n - 1 - i]).abs() < 1e-12,
                "i={i}: {} vs {}",
                gh.weights()[i],
                gh.weights()[n - 1 - i]
            );
        }
    }

    /// Nodes sorted ascending.
    #[test]
    fn nodes_sorted() {
        let gh = GaussHermite::new(20).unwrap();
        for i in 0..gh.order() - 1 {
            assert!(
                gh.nodes()[i] < gh.nodes()[i + 1],
                "i={i}: {} >= {}",
                gh.nodes()[i],
                gh.nodes()[i + 1]
            );
        }
    }

    /// Polynomial exactness: integral of x^(2k) e^(-x^2) = (2k-1)!! sqrt(pi) / 2^k.
    #[test]
    fn polynomial_exactness() {
        let gh = GaussHermite::new(10).unwrap();

        // k=0: integral of 1 * e^(-x^2) = sqrt(pi)
        let r0: f64 = gh.weights().iter().sum();
        assert!((r0 - PI.sqrt()).abs() < 1e-12);

        // k=1: integral of x^2 e^(-x^2) = sqrt(pi)/2
        let r1: f64 = gh
            .nodes()
            .iter()
            .zip(gh.weights())
            .map(|(&x, &w)| x * x * w)
            .sum();
        assert!((r1 - PI.sqrt() / 2.0).abs() < 1e-12, "r1={r1}");

        // k=2: integral of x^4 e^(-x^2) = 3*sqrt(pi)/4
        let r2: f64 = gh
            .nodes()
            .iter()
            .zip(gh.weights())
            .map(|(&x, &w)| x.powi(4) * w)
            .sum();
        assert!((r2 - 3.0 * PI.sqrt() / 4.0).abs() < 1e-11, "r2={r2}");

        // Odd powers should integrate to 0 by symmetry
        let odd: f64 = gh
            .nodes()
            .iter()
            .zip(gh.weights())
            .map(|(&x, &w)| x.powi(3) * w)
            .sum();
        assert!(odd.abs() < 1e-12, "odd={odd}");
    }
}
