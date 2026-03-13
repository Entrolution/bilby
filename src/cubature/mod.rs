//! Multi-dimensional integration (cubature).
//!
//! Provides tensor product rules, sparse grids (Smolyak), adaptive cubature
//! (Genz-Malik), and Monte Carlo / quasi-Monte Carlo methods.
//!
//! # Quick Start
//!
//! ```
//! use bilby::cubature::{TensorProductRule, adaptive_cubature};
//! use bilby::GaussLegendre;
//!
//! // Tensor product: 10-point GL in each of 2 dimensions
//! let gl = GaussLegendre::new(10).unwrap();
//! let tp = TensorProductRule::isotropic(gl.rule(), 2).unwrap();
//! let result = tp.rule().integrate_box(
//!     &[0.0, 0.0], &[1.0, 1.0],
//!     |x| x[0] * x[1],
//! );
//! assert!((result - 0.25).abs() < 1e-14);
//!
//! // Adaptive cubature
//! let result = adaptive_cubature(
//!     |x| (-x[0]*x[0] - x[1]*x[1]).exp(),
//!     &[0.0, 0.0], &[1.0, 1.0], 1e-6,
//! ).unwrap();
//! assert!(result.is_converged());
//! ```

pub mod adaptive;
pub mod halton;
pub mod monte_carlo;
pub mod sobol;
pub mod sparse_grid;
pub mod tensor;

pub use adaptive::{adaptive_cubature, AdaptiveCubature};
pub use monte_carlo::{monte_carlo_integrate, MCMethod, MonteCarloIntegrator};
pub use sparse_grid::{SparseGrid, SparseGridBasis};
pub use tensor::TensorProductRule;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// A precomputed multi-dimensional cubature rule.
///
/// Nodes are stored in a flat layout: node `i` occupies
/// `nodes[i*dim .. (i+1)*dim]`. Weights have length `num_points`.
///
/// The reference domain is \[-1, 1\]^d for deterministic rules
/// (tensor product, sparse grid) and \[0, 1\]^d for Monte Carlo rules.
#[derive(Debug, Clone)]
pub struct CubatureRule {
    /// Nodes stored flat: node i at indices [i*dim .. (i+1)*dim].
    nodes: Vec<f64>,
    /// Weights (length = `num_points`).
    weights: Vec<f64>,
    /// Spatial dimension.
    dim: usize,
}

impl CubatureRule {
    /// Create a new cubature rule from flat node data and weights.
    ///
    /// `nodes_flat` must have length `weights.len() * dim`.
    #[must_use]
    pub fn new(nodes_flat: Vec<f64>, weights: Vec<f64>, dim: usize) -> Self {
        assert_eq!(nodes_flat.len(), weights.len() * dim);
        Self {
            nodes: nodes_flat,
            weights,
            dim,
        }
    }

    /// Number of cubature points.
    #[inline]
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.weights.len()
    }

    /// Spatial dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Access the i-th node as a slice of length `dim`.
    #[inline]
    #[must_use]
    pub fn node(&self, i: usize) -> &[f64] {
        &self.nodes[i * self.dim..(i + 1) * self.dim]
    }

    /// The weights.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Integrate `f` over the reference domain (assumes nodes are on \[-1, 1\]^d).
    #[inline]
    pub fn integrate<G>(&self, f: G) -> f64
    where
        G: Fn(&[f64]) -> f64,
    {
        let mut sum = 0.0;
        for i in 0..self.num_points() {
            sum += self.weights[i] * f(self.node(i));
        }
        sum
    }

    /// Integrate `f` over the hyperrectangle \[lower, upper\].
    ///
    /// Applies an affine transform from the reference domain \[-1, 1\]^d.
    ///
    /// # Panics
    ///
    /// Panics if `lower.len()` or `upper.len()` does not equal the rule's
    /// spatial dimension.
    pub fn integrate_box<G>(&self, lower: &[f64], upper: &[f64], f: G) -> f64
    where
        G: Fn(&[f64]) -> f64,
    {
        assert_eq!(lower.len(), self.dim);
        assert_eq!(upper.len(), self.dim);

        let d = self.dim;
        let half_widths: Vec<f64> = (0..d).map(|j| 0.5 * (upper[j] - lower[j])).collect();
        let midpoints: Vec<f64> = (0..d).map(|j| 0.5 * (lower[j] + upper[j])).collect();
        let jacobian: f64 = half_widths.iter().product();

        let mut x = vec![0.0; d];
        let mut sum = 0.0;
        for i in 0..self.num_points() {
            let node = self.node(i);
            for j in 0..d {
                x[j] = half_widths[j] * node[j] + midpoints[j];
            }
            sum += self.weights[i] * f(&x);
        }
        sum * jacobian
    }

    /// Parallel integration over the reference domain \[-1, 1\]^d.
    ///
    /// Identical to [`integrate`](Self::integrate) but evaluates points in parallel.
    #[cfg(feature = "parallel")]
    pub fn integrate_par<G>(&self, f: G) -> f64
    where
        G: Fn(&[f64]) -> f64 + Sync,
    {
        use rayon::prelude::*;

        let d = self.dim;
        (0..self.num_points())
            .into_par_iter()
            .map(|i| self.weights[i] * f(&self.nodes[i * d..(i + 1) * d]))
            .sum()
    }

    /// Parallel integration over the hyperrectangle \[lower, upper\].
    ///
    /// Identical to [`integrate_box`](Self::integrate_box) but evaluates points in parallel.
    ///
    /// # Panics
    ///
    /// Panics if `lower.len()` or `upper.len()` does not equal the rule's
    /// spatial dimension.
    #[cfg(feature = "parallel")]
    pub fn integrate_box_par<G>(&self, lower: &[f64], upper: &[f64], f: G) -> f64
    where
        G: Fn(&[f64]) -> f64 + Sync,
    {
        use rayon::prelude::*;

        assert_eq!(lower.len(), self.dim);
        assert_eq!(upper.len(), self.dim);

        let d = self.dim;
        let half_widths: Vec<f64> = (0..d).map(|j| 0.5 * (upper[j] - lower[j])).collect();
        let midpoints: Vec<f64> = (0..d).map(|j| 0.5 * (lower[j] + upper[j])).collect();
        let jacobian: f64 = half_widths.iter().product();

        let sum: f64 = (0..self.num_points())
            .into_par_iter()
            .map(|i| {
                let node = &self.nodes[i * d..(i + 1) * d];
                let x: Vec<f64> = (0..d)
                    .map(|j| half_widths[j] * node[j] + midpoints[j])
                    .collect();
                self.weights[i] * f(&x)
            })
            .sum();
        sum * jacobian
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubature_rule_basics() {
        // 1D rule with 2 points (like trapezoidal on [-1,1])
        let rule = CubatureRule::new(vec![-1.0, 1.0], vec![1.0, 1.0], 1);
        assert_eq!(rule.num_points(), 2);
        assert_eq!(rule.dim(), 1);
        assert_eq!(rule.node(0), &[-1.0]);
        assert_eq!(rule.node(1), &[1.0]);
    }

    #[test]
    fn integrate_box_constant() {
        // Single center point in 2D with weight 4 (volume of [-1,1]^2)
        let rule = CubatureRule::new(vec![0.0, 0.0], vec![4.0], 2);
        let result = rule.integrate_box(&[0.0, 0.0], &[2.0, 3.0], |_| 1.0);
        assert!((result - 6.0).abs() < 1e-14); // area = 2*3 = 6
    }

    /// Parallel integrate matches sequential on reference domain.
    #[cfg(feature = "parallel")]
    #[test]
    fn integrate_par_matches_sequential() {
        use crate::cubature::TensorProductRule;
        use crate::GaussLegendre;

        let gl = GaussLegendre::new(10).unwrap();
        let tp = TensorProductRule::isotropic(gl.rule(), 3).unwrap();
        let f = |x: &[f64]| (x[0] * x[1] + x[2]).sin();
        let seq = tp.rule().integrate(&f);
        let par = tp.rule().integrate_par(&f);
        assert!((seq - par).abs() < 1e-14, "seq={seq}, par={par}");
    }

    /// Parallel integrate_box matches sequential.
    #[cfg(feature = "parallel")]
    #[test]
    fn integrate_box_par_matches_sequential() {
        use crate::cubature::TensorProductRule;
        use crate::GaussLegendre;

        let gl = GaussLegendre::new(10).unwrap();
        let tp = TensorProductRule::isotropic(gl.rule(), 2).unwrap();
        let f = |x: &[f64]| x[0] * x[1];
        let seq = tp.rule().integrate_box(&[0.0, 0.0], &[1.0, 1.0], &f);
        let par = tp.rule().integrate_box_par(&[0.0, 0.0], &[1.0, 1.0], &f);
        assert!((seq - par).abs() < 1e-14, "seq={seq}, par={par}");
    }
}
