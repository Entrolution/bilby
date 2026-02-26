//! Gauss-Chebyshev quadrature rules (Types I and II).
//!
//! Nodes and weights have closed-form expressions — no iteration needed.
//!
//! **Type I** — weight function `1 / sqrt(1 - x²)` on \[-1, 1\]:
//!   - Nodes: `x_k = cos((2k - 1) π / (2n))`, k = 1, ..., n
//!   - Weights: `w_k = π / n`
//!
//! **Type II** — weight function `sqrt(1 - x²)` on \[-1, 1\]:
//!   - Nodes: `x_k = cos(k π / (n + 1))`, k = 1, ..., n
//!   - Weights: `w_k = π / (n + 1) * sin²(k π / (n + 1))`

use std::f64::consts::PI;

use crate::error::QuadratureError;
use crate::rule::QuadratureRule;

/// Gauss-Chebyshev Type I quadrature rule.
///
/// Weight function `w(x) = 1 / sqrt(1 - x²)` on \[-1, 1\].
/// To integrate `f(x) / sqrt(1 - x²)`, use this rule directly.
/// To integrate plain `f(x)`, use the underlying [`QuadratureRule`] via [`rule()`](Self::rule).
///
/// # Example
///
/// ```
/// use bilby::GaussChebyshevFirstKind;
///
/// let gc = GaussChebyshevFirstKind::new(10).unwrap();
/// assert_eq!(gc.order(), 10);
/// // All weights equal pi/n for Type I
/// for &w in gc.weights() {
///     assert!((w - std::f64::consts::PI / 10.0).abs() < 1e-14);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GaussChebyshevFirstKind {
    rule: QuadratureRule<f64>,
}

impl GaussChebyshevFirstKind {
    /// Create a new n-point Gauss-Chebyshev Type I rule.
    pub fn new(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }

        let w = PI / n as f64;
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);

        for k in 1..=n {
            let theta = (2 * k - 1) as f64 * PI / (2 * n) as f64;
            nodes.push(theta.cos());
            weights.push(w);
        }

        // Sort nodes ascending (cos decreases as theta increases)
        nodes.reverse();
        weights.reverse();

        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
    }

    /// Returns a reference to the underlying quadrature rule.
    pub fn rule(&self) -> &QuadratureRule<f64> {
        &self.rule
    }

    /// Returns the number of quadrature points.
    pub fn order(&self) -> usize {
        self.rule.order()
    }

    /// Returns the nodes on \[-1, 1\].
    pub fn nodes(&self) -> &[f64] {
        &self.rule.nodes
    }

    /// Returns the weights.
    pub fn weights(&self) -> &[f64] {
        &self.rule.weights
    }
}

/// Gauss-Chebyshev Type II quadrature rule.
///
/// Weight function `w(x) = sqrt(1 - x²)` on \[-1, 1\].
///
/// # Example
///
/// ```
/// use bilby::GaussChebyshevSecondKind;
///
/// let gc = GaussChebyshevSecondKind::new(10).unwrap();
/// assert_eq!(gc.order(), 10);
/// ```
#[derive(Debug, Clone)]
pub struct GaussChebyshevSecondKind {
    rule: QuadratureRule<f64>,
}

impl GaussChebyshevSecondKind {
    /// Create a new n-point Gauss-Chebyshev Type II rule.
    pub fn new(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }

        let n1 = (n + 1) as f64;
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);

        for k in 1..=n {
            let theta = k as f64 * PI / n1;
            nodes.push(theta.cos());
            let s = theta.sin();
            weights.push(PI / n1 * s * s);
        }

        // Sort ascending
        nodes.reverse();
        weights.reverse();

        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
    }

    /// Returns a reference to the underlying quadrature rule.
    pub fn rule(&self) -> &QuadratureRule<f64> {
        &self.rule
    }

    /// Returns the number of quadrature points.
    pub fn order(&self) -> usize {
        self.rule.order()
    }

    /// Returns the nodes on \[-1, 1\].
    pub fn nodes(&self) -> &[f64] {
        &self.rule.nodes
    }

    /// Returns the weights.
    pub fn weights(&self) -> &[f64] {
        &self.rule.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_i_zero_order() {
        assert!(GaussChebyshevFirstKind::new(0).is_err());
    }

    #[test]
    fn type_i_uniform_weights() {
        let gc = GaussChebyshevFirstKind::new(20).unwrap();
        let expected_w = PI / 20.0;
        for &w in gc.weights() {
            assert!((w - expected_w).abs() < 1e-14);
        }
    }

    #[test]
    fn type_i_nodes_sorted_and_bounded() {
        let gc = GaussChebyshevFirstKind::new(50).unwrap();
        for i in 0..gc.order() - 1 {
            assert!(gc.nodes()[i] < gc.nodes()[i + 1]);
        }
        assert!(gc.nodes()[0] > -1.0);
        assert!(*gc.nodes().last().unwrap() < 1.0);
    }

    /// Type I with weight function: integral of 1/sqrt(1-x^2) over [-1,1] = pi.
    /// Sum of weights should equal pi.
    #[test]
    fn type_i_weight_sum() {
        let gc = GaussChebyshevFirstKind::new(100).unwrap();
        let sum: f64 = gc.weights().iter().sum();
        assert!((sum - PI).abs() < 1e-12, "sum={sum}");
    }

    /// Type I exactness: integral of T_k(x) / sqrt(1-x^2) dx over [-1,1].
    /// For T_0(x) = 1, integral = pi. For T_k(x), k > 0, integral = 0.
    /// n-point rule is exact for polynomials up to degree 2n-1.
    #[test]
    fn type_i_polynomial_exactness() {
        let n = 10;
        let gc = GaussChebyshevFirstKind::new(n).unwrap();
        // x^0 weighted integral = pi
        let r: f64 = gc.nodes().iter().zip(gc.weights()).map(|(_, &w)| w).sum();
        assert!((r - PI).abs() < 1e-12);
        // x^1 weighted integral = 0 (by symmetry)
        let r: f64 = gc
            .nodes()
            .iter()
            .zip(gc.weights())
            .map(|(&x, &w)| x * w)
            .sum();
        assert!(r.abs() < 1e-12);
    }

    #[test]
    fn type_ii_zero_order() {
        assert!(GaussChebyshevSecondKind::new(0).is_err());
    }

    /// Type II weight sum: integral of sqrt(1-x^2) over [-1,1] = pi/2.
    #[test]
    fn type_ii_weight_sum() {
        let gc = GaussChebyshevSecondKind::new(100).unwrap();
        let sum: f64 = gc.weights().iter().sum();
        assert!((sum - PI / 2.0).abs() < 1e-12, "sum={sum}");
    }

    #[test]
    fn type_ii_nodes_sorted_and_bounded() {
        let gc = GaussChebyshevSecondKind::new(50).unwrap();
        for i in 0..gc.order() - 1 {
            assert!(gc.nodes()[i] < gc.nodes()[i + 1]);
        }
        assert!(gc.nodes()[0] > -1.0);
        assert!(*gc.nodes().last().unwrap() < 1.0);
    }

    #[test]
    fn type_ii_symmetry() {
        let gc = GaussChebyshevSecondKind::new(21).unwrap();
        let n = gc.order();
        for i in 0..n / 2 {
            assert!((gc.nodes()[i] + gc.nodes()[n - 1 - i]).abs() < 1e-14);
            assert!((gc.weights()[i] - gc.weights()[n - 1 - i]).abs() < 1e-14);
        }
    }
}
