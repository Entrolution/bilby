//! Gauss-Radau quadrature rule.
//!
//! Includes one endpoint in the node set. By default, the left endpoint -1
//! is included (Gauss-Radau-Left). For right endpoint inclusion, use
//! [`GaussRadau::right`].
//!
//! An n-point Gauss-Radau rule is exact for polynomials of degree ≤ 2n-2.
//!
//! Nodes and weights are computed via the Golub-Welsch algorithm with
//! a Radau modification of the Legendre Jacobi matrix.

use crate::error::QuadratureError;
use crate::golub_welsch::{golub_welsch, radau_modify};
use crate::rule::QuadratureRule;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// A Gauss-Radau quadrature rule on \[-1, 1\].
///
/// # Example
///
/// ```
/// use bilby::GaussRadau;
///
/// // Left Radau: includes -1
/// let gr = GaussRadau::left(5).unwrap();
/// assert!((gr.nodes()[0] - (-1.0)).abs() < 1e-14);
///
/// // Right Radau: includes +1
/// let gr = GaussRadau::right(5).unwrap();
/// assert!((gr.nodes()[4] - 1.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct GaussRadau {
    rule: QuadratureRule<f64>,
}

impl GaussRadau {
    /// Create an n-point Gauss-Radau rule including the left endpoint -1.
    ///
    /// Requires `n >= 1`.
    pub fn left(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }
        let (nodes, weights) = compute_radau_left(n);
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
    }

    /// Create an n-point Gauss-Radau rule including the right endpoint +1.
    ///
    /// Requires `n >= 1`.
    pub fn right(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }
        // Right Radau is the reflection of left Radau
        let (nodes_left, weights_left) = compute_radau_left(n);
        let nodes: Vec<f64> = nodes_left.iter().rev().map(|&x| -x).collect();
        let weights: Vec<f64> = weights_left.iter().rev().copied().collect();
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
    }

    /// Returns a reference to the underlying quadrature rule.
    #[inline]
    pub fn rule(&self) -> &QuadratureRule<f64> {
        &self.rule
    }

    /// Returns the number of quadrature points.
    #[inline]
    pub fn order(&self) -> usize {
        self.rule.order()
    }

    /// Returns the nodes on \[-1, 1\].
    #[inline]
    pub fn nodes(&self) -> &[f64] {
        &self.rule.nodes
    }

    /// Returns the weights.
    #[inline]
    pub fn weights(&self) -> &[f64] {
        &self.rule.weights
    }
}

/// Compute n-point left Gauss-Radau nodes and weights (includes -1).
///
/// Uses the Golub-Welsch algorithm with a Radau modification of the
/// Legendre Jacobi matrix. The last diagonal element is modified so
/// that -1 is an eigenvalue of the tridiagonal matrix.
fn compute_radau_left(n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 1 {
        return (vec![-1.0], vec![2.0]);
    }

    // Legendre recurrence coefficients:
    //   α_k = 0 (diagonal)
    //   β_k = k²/(4k²-1) (off-diagonal squared) for k >= 1
    let mut diag = vec![0.0; n];
    let off_diag_sq: Vec<f64> = (1..n)
        .map(|k| {
            let k = k as f64;
            k * k / (4.0 * k * k - 1.0)
        })
        .collect();

    // Modify the last diagonal element so that -1 is an eigenvalue
    radau_modify(&mut diag, &off_diag_sq, -1.0);

    let mu0 = 2.0; // integral of 1 over [-1, 1]
    golub_welsch(&diag, &off_diag_sq, mu0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_order() {
        assert!(GaussRadau::left(0).is_err());
        assert!(GaussRadau::right(0).is_err());
    }

    #[test]
    fn single_point() {
        let gr = GaussRadau::left(1).unwrap();
        assert_eq!(gr.nodes(), &[-1.0]);
        assert!((gr.weights()[0] - 2.0).abs() < 1e-14);
    }

    /// Left Radau includes -1.
    #[test]
    fn left_endpoint() {
        let gr = GaussRadau::left(10).unwrap();
        assert!((gr.nodes()[0] - (-1.0)).abs() < 1e-14);
        assert!(*gr.nodes().last().unwrap() < 1.0);
    }

    /// Right Radau includes +1.
    #[test]
    fn right_endpoint() {
        let gr = GaussRadau::right(10).unwrap();
        assert!(gr.nodes()[0] > -1.0);
        assert!((*gr.nodes().last().unwrap() - 1.0).abs() < 1e-14);
    }

    /// Weight sum = 2.
    #[test]
    fn weight_sum() {
        for n in [2, 5, 10, 20] {
            let gl = GaussRadau::left(n).unwrap();
            let sum: f64 = gl.weights().iter().sum();
            assert!((sum - 2.0).abs() < 1e-12, "n={n}: sum={sum}");

            let gr = GaussRadau::right(n).unwrap();
            let sum: f64 = gr.weights().iter().sum();
            assert!((sum - 2.0).abs() < 1e-12, "right n={n}: sum={sum}");
        }
    }

    /// Nodes sorted ascending.
    #[test]
    fn nodes_sorted() {
        let gr = GaussRadau::left(20).unwrap();
        for i in 0..gr.order() - 1 {
            assert!(
                gr.nodes()[i] < gr.nodes()[i + 1],
                "i={i}: {} >= {}",
                gr.nodes()[i],
                gr.nodes()[i + 1]
            );
        }
    }

    /// Exact for polynomials of degree <= 2n-2.
    #[test]
    fn polynomial_exactness() {
        let n = 10;
        let gr = GaussRadau::left(n).unwrap();
        let max_deg = 2 * n - 2;

        // x^(max_deg) is even, integral = 2/(max_deg+1)
        let expected = 2.0 / (max_deg as f64 + 1.0);
        let result = gr
            .rule()
            .integrate(-1.0, 1.0, |x: f64| x.powi(max_deg as i32));
        assert!(
            (result - expected).abs() < 1e-10,
            "deg={max_deg}: result={result}, expected={expected}"
        );
    }
}
