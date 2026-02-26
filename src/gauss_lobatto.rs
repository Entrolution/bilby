//! Gauss-Lobatto quadrature rule.
//!
//! Includes both endpoints ±1 in the node set. The remaining n-2 interior nodes
//! are roots of `P'_{n-1}(x)` (derivative of the Legendre polynomial).
//!
//! An n-point Gauss-Lobatto rule is exact for polynomials of degree ≤ 2n-3.
//! Important for spectral methods and ODE solvers.

use crate::error::QuadratureError;
use crate::gauss_legendre::legendre_eval;
use crate::rule::QuadratureRule;

/// A Gauss-Lobatto quadrature rule on \[-1, 1\].
///
/// # Example
///
/// ```
/// use bilby::GaussLobatto;
///
/// let gl = GaussLobatto::new(5).unwrap();
/// // Endpoints are exactly -1 and 1
/// assert_eq!(gl.nodes()[0], -1.0);
/// assert_eq!(gl.nodes()[4], 1.0);
///
/// // Integrate x^2 over [-1, 1] = 2/3
/// let result = gl.rule().integrate(-1.0, 1.0, |x: f64| x * x);
/// assert!((result - 2.0 / 3.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct GaussLobatto {
    rule: QuadratureRule<f64>,
}

impl GaussLobatto {
    /// Create a new n-point Gauss-Lobatto rule.
    ///
    /// Requires `n >= 2` (must include both endpoints).
    pub fn new(n: usize) -> Result<Self, QuadratureError> {
        if n < 2 {
            return Err(QuadratureError::InvalidInput(
                "Gauss-Lobatto requires at least 2 points",
            ));
        }

        let (nodes, weights) = compute_lobatto(n);
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

/// Compute n-point Gauss-Lobatto nodes and weights.
///
/// Nodes include -1 and 1. Interior nodes are zeros of P'_{n-1}(x).
/// Weights: w_k = 2 / (n(n-1) [P_{n-1}(x_k)]^2).
fn compute_lobatto(n: usize) -> (Vec<f64>, Vec<f64>) {
    let n_f = n as f64;
    let nm1 = n - 1;
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];

    // Endpoints
    nodes[0] = -1.0;
    nodes[n - 1] = 1.0;
    let w_end = 2.0 / (n_f * (n_f - 1.0));
    weights[0] = w_end;
    weights[n - 1] = w_end;

    if n == 2 {
        return (nodes, weights);
    }

    // Interior nodes: roots of P'_{n-1}(x)
    // P'_{n-1}(x) has n-2 roots in (-1, 1).
    let m = n - 2; // number of interior nodes
    for k in 0..m {
        // Initial guess: Chebyshev-type
        let theta = std::f64::consts::PI * (k as f64 + 1.0) / (m as f64 + 1.0);
        let mut x = -(theta.cos());

        // Newton on P'_{n-1}(x) = 0.
        // P'_{n-1}(x) = (n-1)(x P_{n-1}(x) - P_{n-2}(x)) / (x^2 - 1) ... no, that's complex.
        // Use: the derivative of P_m satisfies a recurrence.
        // Simpler: evaluate P_{n-1}(x) and P'_{n-1}(x), then Newton on P'_{n-1}(x).
        // We need P''_{n-1}(x) for Newton's method applied to P'_{n-1}(x) = 0.
        // P''_m(x) = ((2m-1)*x*P'_m(x) - (m-1+1)*P'_{m-1}(x)) / ... hmm.
        // Use the identity: (1-x^2) P'_n(x) = -n x P_n(x) + n P_{n-1}(x)
        // So P'_n(x) = n(P_{n-1}(x) - x P_n(x)) / (1 - x^2)
        // And we can differentiate again or use:
        // P''_n(x) = (2x P'_n(x) - n(n+1) P_n(x)) / (1 - x^2)
        for _ in 0..100 {
            let (p_nm1, dp_nm1) = legendre_eval(nm1, x);
            // We want the root of dp_nm1 = P'_{n-1}(x) = 0.
            // Need the second derivative P''_{n-1}(x).
            // From (1-x^2) P'_m = m(P_{m-1} - x P_m):
            // Differentiating: -2x P'_m + (1-x^2) P''_m = m(P'_{m-1} - P_m - x P'_m)
            // So P''_m = (2x P'_m + m(P'_{m-1} - P_m - x P'_m)) / (1 - x^2)
            //
            // Simpler approach: use (1-x^2) P''_m(x) = 2x P'_m(x) - m(m+1) P_m(x)
            // Wait, that's for the Legendre equation: (1-x^2)y'' - 2xy' + n(n+1)y = 0
            // So (1-x^2) P''_m(x) = 2x P'_m(x) - m(m+1) P_m(x)
            let nm1_f = nm1 as f64;
            let d2p = (2.0 * x * dp_nm1 - nm1_f * (nm1_f + 1.0) * p_nm1) / (1.0 - x * x);

            if d2p.abs() < 1e-300 {
                break;
            }
            let dx = dp_nm1 / d2p;
            x -= dx;
            if dx.abs() < 1e-15 * (1.0 + x.abs()) {
                break;
            }
        }

        nodes[k + 1] = x;

        // Weight: w_k = 2 / (n(n-1) [P_{n-1}(x_k)]^2)
        let (p_nm1, _) = legendre_eval(nm1, x);
        weights[k + 1] = 2.0 / (n_f * (n_f - 1.0) * p_nm1 * p_nm1);
    }

    // Sort ascending (should already be, but ensure)
    let mut pairs: Vec<_> = nodes.into_iter().zip(weights).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (nodes, weights) = pairs.into_iter().unzip();

    (nodes, weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_points() {
        assert!(GaussLobatto::new(0).is_err());
        assert!(GaussLobatto::new(1).is_err());
        assert!(GaussLobatto::new(2).is_ok());
    }

    #[test]
    fn two_point_trapezoid() {
        let gl = GaussLobatto::new(2).unwrap();
        assert_eq!(gl.nodes(), &[-1.0, 1.0]);
        assert!((gl.weights()[0] - 1.0).abs() < 1e-14);
        assert!((gl.weights()[1] - 1.0).abs() < 1e-14);
    }

    /// Weight sum = 2.
    #[test]
    fn weight_sum() {
        for n in [3, 5, 10, 20, 50] {
            let gl = GaussLobatto::new(n).unwrap();
            let sum: f64 = gl.weights().iter().sum();
            assert!((sum - 2.0).abs() < 1e-12, "n={n}: sum={sum}");
        }
    }

    /// Endpoints are exactly -1 and 1.
    #[test]
    fn endpoints() {
        let gl = GaussLobatto::new(10).unwrap();
        assert_eq!(gl.nodes()[0], -1.0);
        assert_eq!(*gl.nodes().last().unwrap(), 1.0);
    }

    /// Nodes sorted and in [-1, 1].
    #[test]
    fn nodes_sorted() {
        let gl = GaussLobatto::new(20).unwrap();
        for i in 0..gl.order() - 1 {
            assert!(gl.nodes()[i] < gl.nodes()[i + 1]);
        }
    }

    /// Symmetry of nodes and weights.
    #[test]
    fn symmetry() {
        let gl = GaussLobatto::new(15).unwrap();
        let n = gl.order();
        for i in 0..n / 2 {
            assert!(
                (gl.nodes()[i] + gl.nodes()[n - 1 - i]).abs() < 1e-13,
                "i={i}: {} vs {}",
                gl.nodes()[i],
                gl.nodes()[n - 1 - i]
            );
            assert!(
                (gl.weights()[i] - gl.weights()[n - 1 - i]).abs() < 1e-13,
                "i={i}: {} vs {}",
                gl.weights()[i],
                gl.weights()[n - 1 - i]
            );
        }
    }

    /// Exact for polynomials of degree <= 2n-3.
    #[test]
    fn polynomial_exactness() {
        let n = 10;
        let gl = GaussLobatto::new(n).unwrap();
        let max_deg = 2 * n - 3;

        // x^(max_deg) is odd, so integral over [-1,1] = 0
        let result = gl
            .rule()
            .integrate(-1.0, 1.0, |x: f64| x.powi(max_deg as i32));
        assert!(result.abs() < 1e-11, "deg={max_deg}: result={result}");

        // x^(max_deg-1) is even, integral = 2/(max_deg)
        let deg = max_deg - 1;
        let expected = 2.0 / (deg as f64 + 1.0);
        let result = gl.rule().integrate(-1.0, 1.0, |x: f64| x.powi(deg as i32));
        assert!(
            (result - expected).abs() < 1e-11,
            "deg={deg}: result={result}, expected={expected}"
        );
    }
}
