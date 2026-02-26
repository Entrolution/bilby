//! Tensor product cubature rules.
//!
//! Forms the Cartesian product of 1D quadrature rules. For d dimensions
//! with n points each, total cost is n^d — practical for d ≤ 4-5.

use crate::cubature::CubatureRule;
use crate::error::QuadratureError;
use crate::rule::QuadratureRule;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// A tensor-product cubature rule formed from 1D quadrature rules.
///
/// # Example
///
/// ```
/// use bilby::GaussLegendre;
/// use bilby::cubature::TensorProductRule;
///
/// let gl = GaussLegendre::new(5).unwrap();
/// let tp = TensorProductRule::isotropic(gl.rule(), 2).unwrap();
/// assert_eq!(tp.num_points(), 25); // 5^2
///
/// // Integral of x*y over [0,1]^2 = 1/4
/// let result = tp.rule().integrate_box(
///     &[0.0, 0.0], &[1.0, 1.0],
///     |x| x[0] * x[1],
/// );
/// assert!((result - 0.25).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct TensorProductRule {
    rule: CubatureRule,
}

impl TensorProductRule {
    /// Construct from a slice of 1D rules (one per dimension).
    ///
    /// Total points = product of all 1D orders.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if `rules_1d` is empty.
    pub fn new(rules_1d: &[&QuadratureRule<f64>]) -> Result<Self, QuadratureError> {
        if rules_1d.is_empty() {
            return Err(QuadratureError::InvalidInput(
                "at least one dimension required",
            ));
        }

        let dim = rules_1d.len();
        let orders: Vec<usize> = rules_1d.iter().map(|r| r.order()).collect();
        let total_points: usize = orders.iter().product();

        let mut nodes_flat = Vec::with_capacity(total_points * dim);
        let mut weights = Vec::with_capacity(total_points);
        let mut indices = vec![0usize; dim];

        for _ in 0..total_points {
            let mut w = 1.0;
            for j in 0..dim {
                nodes_flat.push(rules_1d[j].nodes[indices[j]]);
                w *= rules_1d[j].weights[indices[j]];
            }
            weights.push(w);

            // Increment multi-index (little-endian)
            for j in 0..dim {
                indices[j] += 1;
                if indices[j] < orders[j] {
                    break;
                }
                indices[j] = 0;
            }
        }

        Ok(Self {
            rule: CubatureRule::new(nodes_flat, weights, dim),
        })
    }

    /// Construct from a single 1D rule applied to all dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if `dim` is zero.
    pub fn isotropic(rule_1d: &QuadratureRule<f64>, dim: usize) -> Result<Self, QuadratureError> {
        let refs: Vec<&QuadratureRule<f64>> = (0..dim).map(|_| rule_1d).collect();
        Self::new(&refs)
    }

    /// Returns a reference to the underlying cubature rule.
    #[inline]
    #[must_use]
    pub fn rule(&self) -> &CubatureRule {
        &self.rule
    }

    /// Number of cubature points.
    #[inline]
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.rule.num_points()
    }

    /// Spatial dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.rule.dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GaussLegendre;

    #[test]
    fn point_count() {
        let gl5 = GaussLegendre::new(5).unwrap();
        let gl3 = GaussLegendre::new(3).unwrap();
        let tp = TensorProductRule::new(&[gl5.rule(), gl3.rule()]).unwrap();
        assert_eq!(tp.num_points(), 15);
        assert_eq!(tp.dim(), 2);
    }

    #[test]
    fn isotropic_point_count() {
        let gl = GaussLegendre::new(4).unwrap();
        let tp = TensorProductRule::isotropic(gl.rule(), 3).unwrap();
        assert_eq!(tp.num_points(), 64); // 4^3
        assert_eq!(tp.dim(), 3);
    }

    /// Weight sum should equal reference domain volume: 2^d for [-1,1]^d.
    #[test]
    fn weight_sum() {
        let gl = GaussLegendre::new(5).unwrap();
        for d in 1..=4 {
            let tp = TensorProductRule::isotropic(gl.rule(), d).unwrap();
            let sum: f64 = tp.rule().weights().iter().sum();
            let expected = 2.0_f64.powi(d as i32);
            assert!(
                (sum - expected).abs() < 1e-12,
                "d={d}: sum={sum}, expected={expected}"
            );
        }
    }

    /// Separable integral: integral of x*y over [0,1]^2 = (1/2)*(1/2) = 1/4.
    #[test]
    fn separable_2d() {
        let gl = GaussLegendre::new(5).unwrap();
        let tp = TensorProductRule::isotropic(gl.rule(), 2).unwrap();
        let result = tp
            .rule()
            .integrate_box(&[0.0, 0.0], &[1.0, 1.0], |x| x[0] * x[1]);
        assert!((result - 0.25).abs() < 1e-14, "result={result}");
    }

    /// Polynomial exactness: 5-point GL is exact for degree 9 per axis.
    #[test]
    fn polynomial_exactness_2d() {
        let gl = GaussLegendre::new(5).unwrap();
        let tp = TensorProductRule::isotropic(gl.rule(), 2).unwrap();
        // integral of x^4 * y^4 over [-1,1]^2 = (2/5)^2
        let result = tp.rule().integrate(|x| x[0].powi(4) * x[1].powi(4));
        let expected = (2.0 / 5.0) * (2.0 / 5.0);
        assert!((result - expected).abs() < 1e-13, "result={result}");
    }

    /// 3D Gaussian integral over [0,1]^3.
    #[test]
    fn gaussian_3d() {
        let gl = GaussLegendre::new(10).unwrap();
        let tp = TensorProductRule::isotropic(gl.rule(), 3).unwrap();
        let result = tp
            .rule()
            .integrate_box(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], |x| {
                (-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])).exp()
            });
        // Reference: (erf(1) * sqrt(pi) / 2)^3 ≈ 0.56370...^3 ≈ 0.17905
        // More precisely: (integral of e^(-x^2) from 0 to 1)^3
        let one_d = 0.7468241328124271; // integral of e^(-x^2) from 0 to 1
        let expected = one_d * one_d * one_d;
        assert!(
            (result - expected).abs() < 1e-10,
            "result={result}, expected={expected}"
        );
    }

    /// Mixed rules: GL(5) in x, GL(3) in y.
    #[test]
    fn mixed_rules() {
        let gl5 = GaussLegendre::new(5).unwrap();
        let gl3 = GaussLegendre::new(3).unwrap();
        let tp = TensorProductRule::new(&[gl5.rule(), gl3.rule()]).unwrap();
        // integral of x^2 + y^2 over [0,1]^2 = 1/3 + 1/3 = 2/3
        let result = tp
            .rule()
            .integrate_box(&[0.0, 0.0], &[1.0, 1.0], |x| x[0] * x[0] + x[1] * x[1]);
        assert!((result - 2.0 / 3.0).abs() < 1e-13, "result={result}");
    }

    #[test]
    fn empty_rules() {
        let result = TensorProductRule::new(&[]);
        assert!(result.is_err());
    }
}
