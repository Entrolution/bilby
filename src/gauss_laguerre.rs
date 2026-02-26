//! Gauss-Laguerre quadrature rule.
//!
//! Weight function `w(x) = x^α e^(-x)` on \[0, ∞).
//!
//! Nodes and weights are computed via the Golub-Welsch algorithm using
//! the generalised Laguerre polynomial three-term recurrence.
//!
//! Standard (non-generalised) Laguerre has α = 0.

use crate::error::QuadratureError;
use crate::gauss_jacobi::ln_gamma;
use crate::golub_welsch::golub_welsch;
use crate::rule::QuadratureRule;

/// A Gauss-Laguerre quadrature rule.
///
/// Integrates `f(x) * x^α * e^(-x)` over \[0, ∞) using n points.
///
/// # Example
///
/// ```
/// use bilby::GaussLaguerre;
///
/// // Standard Laguerre (alpha=0): integral of e^(-x) over [0, inf) = 1
/// let gl = GaussLaguerre::new(10, 0.0).unwrap();
/// let result: f64 = gl.weights().iter().sum();
/// assert!((result - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct GaussLaguerre {
    rule: QuadratureRule<f64>,
    alpha: f64,
}

impl GaussLaguerre {
    /// Create a new n-point Gauss-Laguerre rule with parameter α.
    ///
    /// Requires `n >= 1`, `α > -1`.
    pub fn new(n: usize, alpha: f64) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }
        if alpha <= -1.0 || alpha.is_nan() {
            return Err(QuadratureError::InvalidInput(
                "Laguerre parameter alpha must satisfy alpha > -1",
            ));
        }

        let (nodes, weights) = compute_laguerre(n, alpha);
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
            alpha,
        })
    }

    /// Returns the α parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns a reference to the underlying quadrature rule.
    pub fn rule(&self) -> &QuadratureRule<f64> {
        &self.rule
    }

    /// Returns the number of quadrature points.
    pub fn order(&self) -> usize {
        self.rule.order()
    }

    /// Returns the nodes on \[0, ∞).
    pub fn nodes(&self) -> &[f64] {
        &self.rule.nodes
    }

    /// Returns the weights.
    pub fn weights(&self) -> &[f64] {
        &self.rule.weights
    }
}

/// Compute n Gauss-Laguerre nodes and weights via Golub-Welsch.
///
/// Monic generalised Laguerre recurrence:
///   x L̃_k = L̃_{k+1} + (2k+1+α) L̃_k + k(k+α) L̃_{k-1}
///
/// Jacobi matrix: diagonal = 2k+1+α, off-diagonal² = (k+1)(k+1+α).
/// μ₀ = Γ(α+1).
fn compute_laguerre(n: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let diag: Vec<f64> = (0..n).map(|k| 2.0 * k as f64 + 1.0 + alpha).collect();
    let off_diag_sq: Vec<f64> = (1..n)
        .map(|k| {
            let k = k as f64;
            k * (k + alpha)
        })
        .collect();
    let mu0 = ln_gamma(alpha + 1.0).exp();

    golub_welsch(&diag, &off_diag_sq, mu0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_order() {
        assert!(GaussLaguerre::new(0, 0.0).is_err());
    }

    #[test]
    fn invalid_alpha() {
        assert!(GaussLaguerre::new(5, -1.0).is_err());
        assert!(GaussLaguerre::new(5, -2.0).is_err());
        assert!(GaussLaguerre::new(5, f64::NAN).is_err());
    }

    /// Weight sum for alpha=0: integral of e^(-x) over [0,inf) = Gamma(1) = 1.
    #[test]
    fn standard_weight_sum() {
        let gl = GaussLaguerre::new(20, 0.0).unwrap();
        let sum: f64 = gl.weights().iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum={sum}");
    }

    /// Weight sum for alpha: integral of x^alpha e^(-x) over [0,inf) = Gamma(alpha+1).
    #[test]
    fn generalised_weight_sum() {
        for alpha in [0.5, 1.0, 2.0, 3.5] {
            let gl = GaussLaguerre::new(20, alpha).unwrap();
            let sum: f64 = gl.weights().iter().sum();
            let expected = ln_gamma(alpha + 1.0).exp();
            assert!(
                (sum - expected).abs() < 1e-10,
                "alpha={alpha}: sum={sum}, expected={expected}"
            );
        }
    }

    /// Nodes should be positive and sorted ascending.
    #[test]
    fn nodes_positive_and_sorted() {
        let gl = GaussLaguerre::new(20, 0.0).unwrap();
        for &x in gl.nodes() {
            assert!(x > 0.0, "node={x} is not positive");
        }
        for i in 0..gl.order() - 1 {
            assert!(gl.nodes()[i] < gl.nodes()[i + 1]);
        }
    }

    /// Polynomial exactness: integral of x^k e^(-x) over [0,inf) = k! = Gamma(k+1).
    #[test]
    fn polynomial_exactness() {
        let n = 10;
        let gl = GaussLaguerre::new(n, 0.0).unwrap();
        for k in 0..5 {
            let numerical: f64 = gl
                .nodes()
                .iter()
                .zip(gl.weights())
                .map(|(&x, &w)| w * x.powi(k))
                .sum();
            let expected = ln_gamma(k as f64 + 1.0).exp();
            assert!(
                (numerical - expected).abs() < 1e-10,
                "k={k}: got={numerical}, expected={expected}"
            );
        }
    }
}
