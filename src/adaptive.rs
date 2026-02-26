//! Adaptive quadrature via global subdivision.
//!
//! Implements QUADPACK-style adaptive integration using Gauss-Kronrod pairs
//! for error estimation. The algorithm maintains a priority queue of subintervals
//! ordered by error estimate, repeatedly bisecting the worst interval until the
//! global error target is met or the evaluation budget is exhausted.
//!
//! # Example
//!
//! ```
//! use bilby::adaptive_integrate;
//!
//! let result = adaptive_integrate(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 1e-10).unwrap();
//! assert!((result.value - 2.0).abs() < 1e-10);
//! assert!(result.is_converged());
//! ```

use core::cmp::Ordering;

#[cfg(not(feature = "std"))]
use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(feature = "std")]
use std::collections::BinaryHeap;

use crate::error::QuadratureError;
use crate::gauss_kronrod::{GKPair, GaussKronrod};
use crate::result::QuadratureResult;

/// A subinterval with its integration estimate and error.
#[derive(Debug, Clone)]
struct Subinterval {
    a: f64,
    b: f64,
    estimate: f64,
    error: f64,
}

impl PartialEq for Subinterval {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl Eq for Subinterval {}

impl PartialOrd for Subinterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Subinterval {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by error. NaN sorts low.
        self.error
            .partial_cmp(&other.error)
            .unwrap_or(Ordering::Equal)
    }
}

/// Builder for configuring adaptive integration.
///
/// # Example
///
/// ```
/// use bilby::{AdaptiveIntegrator, GKPair};
///
/// let result = AdaptiveIntegrator::default()
///     .with_pair(GKPair::G10K21)
///     .with_abs_tol(1e-12)
///     .with_rel_tol(1e-12)
///     .integrate(0.0, 1.0, |x: f64| x.exp())
///     .unwrap();
///
/// assert!((result.value - (1.0_f64.exp() - 1.0)).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveIntegrator {
    pair: GKPair,
    abs_tol: f64,
    rel_tol: f64,
    max_evals: usize,
}

impl Default for AdaptiveIntegrator {
    fn default() -> Self {
        Self {
            pair: GKPair::G7K15,
            abs_tol: 1.49e-8,
            rel_tol: 1.49e-8,
            max_evals: 10_000,
        }
    }
}

impl AdaptiveIntegrator {
    /// Set the Gauss-Kronrod pair used for panel evaluation.
    pub fn with_pair(mut self, pair: GKPair) -> Self {
        self.pair = pair;
        self
    }

    /// Set the absolute error tolerance.
    pub fn with_abs_tol(mut self, tol: f64) -> Self {
        self.abs_tol = tol;
        self
    }

    /// Set the relative error tolerance.
    pub fn with_rel_tol(mut self, tol: f64) -> Self {
        self.rel_tol = tol;
        self
    }

    /// Set the maximum number of function evaluations.
    pub fn with_max_evals(mut self, n: usize) -> Self {
        self.max_evals = n;
        self
    }

    /// Adaptively integrate `f` over \[a, b\].
    ///
    /// Returns a [`QuadratureResult`] with the best estimate and error bound.
    /// If the tolerance could not be achieved within the evaluation budget,
    /// `converged` will be `false` but the best estimate is still returned.
    pub fn integrate<G>(
        &self,
        a: f64,
        b: f64,
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(f64) -> f64,
    {
        self.integrate_intervals(&[(a, b)], &f)
    }

    /// Adaptively integrate `f` over \[a, b\] with known break points.
    ///
    /// Break points split the interval at known discontinuities or singularities.
    /// The adaptive algorithm then refines each sub-interval independently in a
    /// global priority queue.
    pub fn integrate_with_breaks<G>(
        &self,
        a: f64,
        b: f64,
        breaks: &[f64],
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(f64) -> f64,
    {
        validate_breaks(a, b, breaks)?;
        let intervals = build_intervals(a, b, breaks);
        self.integrate_intervals(&intervals, &f)
    }

    /// Core adaptive loop over a set of initial intervals.
    fn integrate_intervals<G>(
        &self,
        intervals: &[(f64, f64)],
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(f64) -> f64,
    {
        // Validate tolerances
        if self.abs_tol <= 0.0 && self.rel_tol <= 0.0 {
            return Err(QuadratureError::InvalidInput(
                "at least one tolerance must be positive",
            ));
        }

        let gk = GaussKronrod::new(self.pair);
        let evals_per_call = self.pair.kronrod_order();

        if self.max_evals < evals_per_call {
            return Err(QuadratureError::InvalidInput(
                "max_evals is less than one GK panel evaluation",
            ));
        }

        let mut heap: BinaryHeap<Subinterval> = BinaryHeap::new();
        let mut total_estimate = 0.0;
        let mut total_error = 0.0;
        let mut num_evals = 0;

        // Seed the priority queue with initial intervals
        for &(ia, ib) in intervals {
            if ia == ib {
                continue;
            }
            let detail = gk.integrate_detail(ia, ib, f);
            num_evals += evals_per_call;

            total_estimate += detail.estimate;
            total_error += detail.error;

            heap.push(Subinterval {
                a: ia,
                b: ib,
                estimate: detail.estimate,
                error: detail.error,
            });
        }

        // Adaptive refinement loop
        while !self.tolerance_met(total_estimate, total_error)
            && num_evals + 2 * evals_per_call <= self.max_evals
        {
            let worst = match heap.pop() {
                Some(s) => s,
                None => break,
            };

            // Bisect the worst interval
            let mid = 0.5 * (worst.a + worst.b);

            let left = gk.integrate_detail(worst.a, mid, f);
            let right = gk.integrate_detail(mid, worst.b, f);
            num_evals += 2 * evals_per_call;

            // Update totals: remove old contribution, add new
            total_estimate = total_estimate - worst.estimate + left.estimate + right.estimate;
            total_error = total_error - worst.error + left.error + right.error;

            heap.push(Subinterval {
                a: worst.a,
                b: mid,
                estimate: left.estimate,
                error: left.error,
            });
            heap.push(Subinterval {
                a: mid,
                b: worst.b,
                estimate: right.estimate,
                error: right.error,
            });
        }

        let converged = self.tolerance_met(total_estimate, total_error);

        Ok(QuadratureResult {
            value: total_estimate,
            error_estimate: total_error,
            num_evals,
            converged,
        })
    }

    /// Check whether the current error is within tolerance.
    fn tolerance_met(&self, estimate: f64, error: f64) -> bool {
        error <= self.abs_tol.max(self.rel_tol * estimate.abs())
    }
}

/// Validate break points are within (a, b) and not NaN.
fn validate_breaks(a: f64, b: f64, breaks: &[f64]) -> Result<(), QuadratureError> {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    for &bp in breaks {
        if bp.is_nan() {
            return Err(QuadratureError::InvalidInput("break point is NaN"));
        }
        if bp <= lo || bp >= hi {
            return Err(QuadratureError::InvalidInput(
                "break point outside interval (a, b)",
            ));
        }
    }
    Ok(())
}

/// Build sorted, deduplicated intervals from break points.
fn build_intervals(a: f64, b: f64, breaks: &[f64]) -> Vec<(f64, f64)> {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let mut points = vec![lo];
    for &bp in breaks {
        if bp > lo && bp < hi {
            points.push(bp);
        }
    }
    points.push(hi);
    points.sort_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
    points.dedup();

    let mut intervals = Vec::with_capacity(points.len() - 1);
    for w in points.windows(2) {
        intervals.push((w[0], w[1]));
    }

    // If a > b, the user wants the negated integral — reverse the intervals
    // and the adaptive integrator will produce a negated result naturally
    // via the GK rule's affine transform.
    if a > b {
        intervals = intervals
            .into_iter()
            .map(|(lo, hi)| (hi, lo))
            .rev()
            .collect();
    }

    intervals
}

/// Adaptively integrate `f` over \[a, b\] with default settings.
///
/// Uses G7-K15 with absolute and relative tolerance both set to `tol`.
///
/// # Errors
///
/// Returns [`QuadratureError::InvalidInput`] if `tol` is not positive.
///
/// # Example
///
/// ```
/// use bilby::adaptive_integrate;
///
/// let result = adaptive_integrate(|x: f64| x.exp(), 0.0, 1.0, 1e-10).unwrap();
/// assert!((result.value - (1.0_f64.exp() - 1.0)).abs() < 1e-10);
/// ```
pub fn adaptive_integrate<G>(
    f: G,
    a: f64,
    b: f64,
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    AdaptiveIntegrator::default()
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .integrate(a, b, f)
}

/// Adaptively integrate `f` over \[a, b\] with known break points.
///
/// Break points indicate locations of discontinuities or singularities.
/// The interval is split at these points before adaptive refinement begins.
///
/// # Errors
///
/// Returns [`QuadratureError::InvalidInput`] if any break point is outside `(a, b)` or NaN.
pub fn adaptive_integrate_with_breaks<G>(
    f: G,
    a: f64,
    b: f64,
    breaks: &[f64],
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    AdaptiveIntegrator::default()
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .integrate_with_breaks(a, b, breaks, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integral of x^2 over [0, 1] = 1/3.
    #[test]
    fn polynomial() {
        let r = adaptive_integrate(|x| x * x, 0.0, 1.0, 1e-12).unwrap();
        assert!(r.converged, "did not converge: err={}", r.error_estimate);
        assert!(
            (r.value - 1.0 / 3.0).abs() < 1e-12,
            "value={}, expected 1/3",
            r.value
        );
    }

    /// Integral of sin(x) over [0, pi] = 2.
    #[test]
    fn sin_integral() {
        let r = adaptive_integrate(f64::sin, 0.0, core::f64::consts::PI, 1e-12).unwrap();
        assert!(r.converged);
        assert!((r.value - 2.0).abs() < 1e-12, "value={}", r.value);
    }

    /// Integral of exp(x) over [0, 1] = e - 1.
    #[test]
    fn exp_integral() {
        let expected = 1.0_f64.exp() - 1.0;
        let r = adaptive_integrate(f64::exp, 0.0, 1.0, 1e-12).unwrap();
        assert!(r.converged);
        assert!((r.value - expected).abs() < 1e-12, "value={}", r.value);
    }

    /// Reversed bounds should negate the result.
    #[test]
    fn reversed_bounds() {
        let forward = adaptive_integrate(f64::sin, 0.0, core::f64::consts::PI, 1e-12).unwrap();
        let reverse = adaptive_integrate(f64::sin, core::f64::consts::PI, 0.0, 1e-12).unwrap();
        assert!(
            (forward.value + reverse.value).abs() < 1e-12,
            "forward={}, reverse={}",
            forward.value,
            reverse.value
        );
    }

    /// a == b gives zero.
    #[test]
    fn zero_width() {
        let r = adaptive_integrate(f64::sin, 1.0, 1.0, 1e-12).unwrap();
        assert_eq!(r.value, 0.0);
        assert!(r.converged);
    }

    /// Invalid tolerance.
    #[test]
    fn invalid_tolerance() {
        let r = AdaptiveIntegrator::default()
            .with_abs_tol(0.0)
            .with_rel_tol(0.0)
            .integrate(0.0, 1.0, f64::sin);
        assert!(r.is_err());
    }

    /// Integration with break points: |x| over [-1, 1] = 1 with break at 0.
    #[test]
    fn absolute_value_with_break() {
        let r = adaptive_integrate_with_breaks(|x: f64| x.abs(), -1.0, 1.0, &[0.0], 1e-12).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!((r.value - 1.0).abs() < 1e-12, "value={}", r.value);
    }

    /// Step function with break point.
    #[test]
    fn step_function_with_break() {
        let step = |x: f64| if x < 2.0 { 1.0 } else { 3.0 };
        let r = adaptive_integrate_with_breaks(step, 0.0, 4.0, &[2.0], 1e-12).unwrap();
        let expected = 1.0 * 2.0 + 3.0 * 2.0; // 8.0
        assert!(r.converged, "err={}", r.error_estimate);
        assert!((r.value - expected).abs() < 1e-12, "value={}", r.value);
    }

    /// Break point outside interval is rejected.
    #[test]
    fn break_outside_interval() {
        let r = adaptive_integrate_with_breaks(f64::sin, 0.0, 1.0, &[2.0], 1e-12);
        assert!(r.is_err());
    }

    /// Builder with higher-order pair achieves tight tolerance.
    #[test]
    fn builder_high_order() {
        let r = AdaptiveIntegrator::default()
            .with_pair(GKPair::G15K31)
            .with_abs_tol(1e-14)
            .with_rel_tol(1e-14)
            .with_max_evals(100_000)
            .integrate(0.0, core::f64::consts::PI, f64::sin)
            .unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!((r.value - 2.0).abs() < 1e-14, "value={}", r.value);
    }

    /// Non-convergence within tight budget is signalled via converged flag.
    #[test]
    fn non_convergence() {
        // Highly oscillatory function with very few evals allowed
        let r = AdaptiveIntegrator::default()
            .with_abs_tol(1e-15)
            .with_rel_tol(1e-15)
            .with_max_evals(15) // Only one GK15 evaluation
            .integrate(0.0, 100.0, |x: f64| (100.0 * x).sin())
            .unwrap();
        assert!(!r.converged);
        // Still returns a best estimate
        assert!(r.value.is_finite());
    }

    /// Peaked function requires many subdivisions.
    #[test]
    fn peaked_function() {
        // 1 / (1 + (x - 0.3)^2 * 1e4) over [0, 1]
        // Known integral: pi/100 * (atan(70) + atan(30)) ≈ 0.0313...
        let r = AdaptiveIntegrator::default()
            .with_pair(GKPair::G10K21)
            .with_abs_tol(1e-8)
            .with_rel_tol(1e-8)
            .with_max_evals(50_000)
            .integrate(0.0, 1.0, |x| 1.0 / (1.0 + (x - 0.3).powi(2) * 1e4))
            .unwrap();
        let expected = (70.0_f64.atan() + 30.0_f64.atan()) / 100.0;
        assert!(r.converged, "err={}", r.error_estimate);
        assert!(
            (r.value - expected).abs() < 1e-8,
            "value={}, expected={}",
            r.value,
            expected
        );
    }

    /// Tracks evaluation count correctly.
    #[test]
    fn eval_count() {
        let r = adaptive_integrate(|x| x * x, 0.0, 1.0, 1e-12).unwrap();
        // Should converge quickly for a polynomial — just a few GK15 evals
        assert!(r.num_evals > 0);
        assert!(r.num_evals <= 200, "num_evals={}", r.num_evals);
    }
}
