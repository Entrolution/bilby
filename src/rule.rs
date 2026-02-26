//! Core quadrature rule types.

use num_traits::Float;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A precomputed quadrature rule on the reference interval \[-1, 1\].
///
/// Stores nodes and weights that can be reused across many integrations.
/// Rule construction allocates; integration with a precomputed rule does not.
#[derive(Debug, Clone)]
pub struct QuadratureRule<F> {
    /// Quadrature nodes on \[-1, 1\].
    pub nodes: Vec<F>,
    /// Corresponding quadrature weights.
    pub weights: Vec<F>,
}

impl<F: Float> QuadratureRule<F> {
    /// Returns the number of points in this rule.
    #[inline]
    #[must_use]
    pub fn order(&self) -> usize {
        self.nodes.len()
    }

    /// Integrate `f` over \[a, b\] using this rule.
    ///
    /// Applies an affine transform from \[-1, 1\] to \[a, b\]:
    ///   x = (b - a) / 2 * t + (a + b) / 2
    ///
    /// The result is scaled by (b - a) / 2.
    ///
    /// # Panics
    ///
    /// Panics if the conversion of `2.0` to `F` via `F::from` fails, which
    /// should not occur for standard floating-point types.
    #[inline]
    pub fn integrate<G>(&self, a: F, b: F, f: G) -> F
    where
        G: Fn(F) -> F,
    {
        let two = F::from(2.0).unwrap();
        let half_width = (b - a) / two;
        let midpoint = (a + b) / two;

        let mut sum = F::zero();
        for (node, weight) in self.nodes.iter().zip(self.weights.iter()) {
            let x = half_width * *node + midpoint;
            sum = sum + *weight * f(x);
        }
        sum * half_width
    }

    /// Integrate `f` over \[a, b\] using a composite rule with `n_panels` equal subintervals.
    ///
    /// Each panel applies this quadrature rule independently, then sums the results.
    /// Total function evaluations = `self.order() * n_panels`.
    ///
    /// # Panics
    ///
    /// Panics if `n_panels` or the panel indices cannot be converted to `F`
    /// via `F::from`, which should not occur for standard floating-point types
    /// with reasonable panel counts.
    pub fn integrate_composite<G>(&self, a: F, b: F, n_panels: usize, f: G) -> F
    where
        G: Fn(F) -> F,
    {
        let n = F::from(n_panels).unwrap();
        let panel_width = (b - a) / n;
        let mut total = F::zero();

        for i in 0..n_panels {
            let panel_a = a + F::from(i).unwrap() * panel_width;
            let panel_b = panel_a + panel_width;
            total = total + self.integrate(panel_a, panel_b, &f);
        }
        total
    }

    /// Parallel composite integration over \[a, b\] with `n_panels` subintervals.
    ///
    /// Identical to [`integrate_composite`](Self::integrate_composite) but evaluates
    /// panels in parallel using rayon. Requires `F: Send + Sync` and `f: Fn(F) -> F + Sync`.
    ///
    /// # Panics
    ///
    /// Panics if `n_panels` or the panel indices cannot be converted to `F`
    /// via `F::from`, which should not occur for standard floating-point types
    /// with reasonable panel counts.
    #[cfg(feature = "parallel")]
    pub fn integrate_composite_par<G>(&self, a: F, b: F, n_panels: usize, f: G) -> F
    where
        F: Send + Sync,
        G: Fn(F) -> F + Sync,
    {
        use rayon::prelude::*;

        let n = F::from(n_panels).unwrap();
        let panel_width = (b - a) / n;

        (0..n_panels)
            .into_par_iter()
            .map(|i| {
                let panel_a = a + F::from(i).unwrap() * panel_width;
                let panel_b = panel_a + panel_width;
                self.integrate(panel_a, panel_b, &f)
            })
            .reduce(F::zero, |a, b| a + b)
    }
}
