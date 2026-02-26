//! Clenshaw-Curtis quadrature rule.
//!
//! Nodes are Chebyshev points (extrema of Chebyshev polynomials).
//! An n-point rule uses `x_k = cos(k π / (n-1))` for k = 0, ..., n-1.
//!
//! Weights are computed via an explicit formula based on the discrete cosine
//! transform structure.
//!
//! **Key property**: nested — doubling n reuses all previous nodes, which
//! makes Clenshaw-Curtis attractive for adaptive refinement.

use crate::error::QuadratureError;
use crate::rule::QuadratureRule;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// A Clenshaw-Curtis quadrature rule on \[-1, 1\].
///
/// # Example
///
/// ```
/// use bilby::ClenshawCurtis;
///
/// let cc = ClenshawCurtis::new(11).unwrap();
/// // Integrate x^4 over [-1, 1] = 2/5
/// let result = cc.rule().integrate(-1.0, 1.0, |x: f64| x.powi(4));
/// assert!((result - 0.4).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct ClenshawCurtis {
    rule: QuadratureRule<f64>,
}

impl ClenshawCurtis {
    /// Create a new n-point Clenshaw-Curtis rule.
    ///
    /// Requires `n >= 1`. For `n == 1`, returns the midpoint rule.
    pub fn new(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }

        let (nodes, weights) = compute_clenshaw_curtis(n);
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

/// Compute n-point Clenshaw-Curtis nodes and weights.
///
/// For n == 1: midpoint rule (x=0, w=2).
/// For n >= 2: nodes at x_k = cos(k π / (n-1)), weights via explicit formula.
fn compute_clenshaw_curtis(n: usize) -> (Vec<f64>, Vec<f64>) {
    use core::f64::consts::PI;

    if n == 1 {
        return (vec![0.0], vec![2.0]);
    }

    let nm1 = n - 1;
    let nm1_f = nm1 as f64;

    // Nodes: x_k = cos(k * pi / (n-1)), k = 0, ..., n-1
    // These go from 1 to -1, so reverse for ascending order.
    let mut nodes = Vec::with_capacity(n);
    for k in 0..n {
        nodes.push((k as f64 * PI / nm1_f).cos());
    }
    nodes.reverse(); // ascending: -1 to 1

    // Weights via the explicit formula.
    // w_k = c_k / (n-1) * sum_{j=0}^{floor((n-1)/2)} b_j / (1 - 4j^2) * cos(2jk pi/(n-1))
    // where c_k = 1 if k=0 or k=n-1, else 2; and b_j = 1 if j=0 or j=(n-1)/2, else 2.
    //
    // Reversed index since we reversed nodes.
    let mut weights = vec![0.0_f64; n];
    let m = nm1 / 2; // floor((n-1)/2)

    for (i, w) in weights.iter_mut().enumerate() {
        // Original index (before reversal)
        let k = nm1 - i;
        let mut sum = 0.0;
        for j in 0..=m {
            let j_f = j as f64;
            let b_j = if j == 0 || (nm1.is_multiple_of(2) && j == m) {
                1.0
            } else {
                2.0
            };
            let denom = 1.0 - 4.0 * j_f * j_f;
            sum += b_j / denom * (2.0 * j_f * k as f64 * PI / nm1_f).cos();
        }
        let c_k = if k == 0 || k == nm1 { 1.0 } else { 2.0 };
        *w = c_k * sum / nm1_f;
    }

    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_order() {
        assert!(ClenshawCurtis::new(0).is_err());
    }

    #[test]
    fn single_point() {
        let cc = ClenshawCurtis::new(1).unwrap();
        assert_eq!(cc.nodes(), &[0.0]);
        assert!((cc.weights()[0] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn two_points() {
        let cc = ClenshawCurtis::new(2).unwrap();
        assert_eq!(cc.nodes()[0], -1.0);
        assert_eq!(cc.nodes()[1], 1.0);
        assert!((cc.weights()[0] - 1.0).abs() < 1e-14);
        assert!((cc.weights()[1] - 1.0).abs() < 1e-14);
    }

    /// Weight sum = 2.
    #[test]
    fn weight_sum() {
        for n in [3, 5, 11, 21, 51] {
            let cc = ClenshawCurtis::new(n).unwrap();
            let sum: f64 = cc.weights().iter().sum();
            assert!((sum - 2.0).abs() < 1e-12, "n={n}: sum={sum}");
        }
    }

    /// Endpoints should be exactly -1 and 1 for n >= 2.
    #[test]
    fn endpoints() {
        let cc = ClenshawCurtis::new(11).unwrap();
        assert_eq!(cc.nodes()[0], -1.0);
        assert_eq!(*cc.nodes().last().unwrap(), 1.0);
    }

    /// Nodes sorted ascending.
    #[test]
    fn nodes_sorted() {
        let cc = ClenshawCurtis::new(21).unwrap();
        for i in 0..cc.order() - 1 {
            assert!(cc.nodes()[i] < cc.nodes()[i + 1]);
        }
    }

    /// Symmetry of nodes and weights.
    #[test]
    fn symmetry() {
        let cc = ClenshawCurtis::new(21).unwrap();
        let n = cc.order();
        for i in 0..n / 2 {
            assert!(
                (cc.nodes()[i] + cc.nodes()[n - 1 - i]).abs() < 1e-14,
                "i={i}: {} vs {}",
                cc.nodes()[i],
                cc.nodes()[n - 1 - i]
            );
            assert!(
                (cc.weights()[i] - cc.weights()[n - 1 - i]).abs() < 1e-14,
                "i={i}: {} vs {}",
                cc.weights()[i],
                cc.weights()[n - 1 - i]
            );
        }
    }

    /// n-point Clenshaw-Curtis is exact for polynomials of degree <= n-1.
    #[test]
    fn polynomial_exactness() {
        let n = 11;
        let cc = ClenshawCurtis::new(n).unwrap();

        // x^(n-1) is even (n-1=10), integral = 2/11
        let deg = n - 1;
        let expected = 2.0 / (deg as f64 + 1.0);
        let result = cc.rule().integrate(-1.0, 1.0, |x: f64| x.powi(deg as i32));
        assert!(
            (result - expected).abs() < 1e-12,
            "deg={deg}: result={result}, expected={expected}"
        );
    }

    /// Integrate sin(x) over [0, pi] = 2.
    #[test]
    fn sin_integration() {
        let cc = ClenshawCurtis::new(21).unwrap();
        let result = cc.rule().integrate(0.0, core::f64::consts::PI, f64::sin);
        assert!((result - 2.0).abs() < 1e-12, "result={result}");
    }

    /// Nesting: all nodes of CC(n) appear in CC(2n-1).
    #[test]
    fn nesting() {
        let cc5 = ClenshawCurtis::new(5).unwrap();
        let cc9 = ClenshawCurtis::new(9).unwrap();

        for &x5 in cc5.nodes() {
            let found = cc9.nodes().iter().any(|&x9| (x5 - x9).abs() < 1e-14);
            assert!(found, "node {x5} from CC(5) not found in CC(9)");
        }
    }
}
