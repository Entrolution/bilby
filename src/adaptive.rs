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
        // Max-heap by error, as a total order including NaN. The previous
        // `partial_cmp(...).unwrap_or(Equal)` made a NaN error compare equal to
        // every value, which is non-transitive; total_cmp is a proper total order.
        self.error.total_cmp(&other.error)
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
    #[must_use]
    pub fn with_pair(mut self, pair: GKPair) -> Self {
        self.pair = pair;
        self
    }

    /// Set the absolute error tolerance.
    #[must_use]
    pub fn with_abs_tol(mut self, tol: f64) -> Self {
        self.abs_tol = tol;
        self
    }

    /// Set the relative error tolerance.
    #[must_use]
    pub fn with_rel_tol(mut self, tol: f64) -> Self {
        self.rel_tol = tol;
        self
    }

    /// Set the maximum number of function evaluations.
    #[must_use]
    pub fn with_max_evals(mut self, n: usize) -> Self {
        self.max_evals = n;
        self
    }

    /// Adaptively integrate `f` over \[a, b\].
    ///
    /// Returns a [`QuadratureResult`] with the best estimate and error bound.
    /// If the tolerance could not be achieved, `converged` will be `false` but
    /// the best estimate is still returned; in that case
    /// [`roundoff_limited`](QuadratureResult::roundoff_limited) distinguishes
    /// reaching the floating-point floor (the tolerance was unachievable in
    /// `f64`) from exhausting the evaluation budget.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if both tolerances are non-positive
    /// or if `max_evals` is less than one Gauss-Kronrod panel evaluation.
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
    /// global priority queue. As with [`integrate`](Self::integrate), the result
    /// reports [`converged`](QuadratureResult::converged) and
    /// [`roundoff_limited`](QuadratureResult::roundoff_limited) so a tolerance
    /// that is unachievable in `f64` is distinguishable from budget exhaustion.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if any break point is outside
    /// `(a, b)`, is NaN, or if both tolerances are non-positive.
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
        // Validate interval bounds
        for &(a, b) in intervals {
            if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
                return Err(QuadratureError::DegenerateInterval);
            }
        }

        // Validate tolerances. A NaN tolerance would pass the `<= 0.0` check
        // (NaN compares false) and then make every `error <= tol` test false,
        // spinning to max_evals instead of erroring.
        if self.abs_tol.is_nan() || self.rel_tol.is_nan() {
            return Err(QuadratureError::InvalidInput("tolerances must not be NaN"));
        }
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

        // Seeding every (non-degenerate) break-point subinterval must fit within
        // the eval budget; otherwise the documented max_evals contract would be
        // silently overrun on the integrate_with_breaks path. Compared via
        // division to avoid overflow in n_seed * evals_per_call.
        #[allow(clippy::float_cmp)]
        let n_seed = intervals.iter().filter(|(ia, ib)| ia != ib).count();
        if n_seed > self.max_evals / evals_per_call {
            return Err(QuadratureError::InvalidInput(
                "max_evals is too small to seed all break-point subintervals",
            ));
        }

        // Seed the priority queue with initial intervals
        for &(ia, ib) in intervals {
            // Exact comparison is intentional: degenerate zero-width
            // intervals are an edge case, not a floating-point tolerance issue.
            #[allow(clippy::float_cmp)]
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

        // Adaptive refinement loop. `roundoff_count` tracks consecutive panels
        // whose bisection neither reduced the error nor moved the integral (and
        // were not merely under-resolved) — the QUADPACK ier=2 roundoff signal;
        // after ROUNDOFF_STALL_LIMIT of them the floating-point floor is reached.
        // QUADPACK's dqagse accumulates iroff1/iroff2/iroff3 across the whole
        // run (flagging at 10/20); the consecutive-with-reset policy here is a
        // deliberately more conservative simplification — any genuinely
        // improving bisection resets the count, so a flag requires an unbroken
        // run of floor-level steps. `no_error_reduction` also folds in dqagse's
        // iroff3 "error failed to decrease" case.
        const ROUNDOFF_STALL_LIMIT: usize = 6;
        let mut roundoff_count = 0usize;
        let mut roundoff_limited = false;
        while !self.tolerance_met(total_estimate, total_error)
            && num_evals + 2 * evals_per_call <= self.max_evals
        {
            let Some(worst) = heap.pop() else { break };

            // Bisect the worst interval
            let mid = 0.5 * (worst.a + worst.b);

            // If the midpoint rounds to an endpoint the interval has shrunk to
            // the floating-point resolution limit (typically at an integrable
            // singularity); bisecting it cannot reduce the error. Drop it and
            // refine the next-worst interval rather than spinning to max_evals
            // re-evaluating the same panel. Exact equality is orientation-
            // independent, so this also holds for reversed bounds (a > b). Its
            // error stays in total_error, so convergence is not falsely declared.
            // A drop is itself an at-resolution-limit event: it leaves
            // `roundoff_count` untouched (neither incremented nor reset), so the
            // consecutive-stall run is transparent to interleaved drops — which
            // is benign, since a drop cannot manufacture the increment conditions.
            #[allow(clippy::float_cmp)]
            if mid == worst.a || mid == worst.b {
                continue;
            }

            let left = gk.integrate_detail(worst.a, mid, f);
            let right = gk.integrate_detail(mid, worst.b, f);
            num_evals += 2 * evals_per_call;

            // QUADPACK dqagse roundoff-stall detection (ier=2): a bisection that
            // neither reduced the error (children error >= 0.99 * parent) nor
            // moved the integral (area stable) signals the floating-point floor
            // — UNLESS a child's error merely saturated its resasc cap
            // (`error == resasc`), which means the panel is under-resolved, not
            // floor-limited, and refinement must continue. Counting only
            // consecutive such events avoids a false stall on a hard integrand.
            let children_estimate = left.estimate + right.estimate;
            let children_error = left.error + right.error;
            #[allow(clippy::float_cmp)]
            let child_saturated = left.error == left.resasc || right.error == right.resasc;
            let no_error_reduction = children_error >= worst.error * 0.99;
            let area_stable =
                (children_estimate - worst.estimate).abs() <= 1.0e-5 * children_estimate.abs();
            if no_error_reduction && area_stable && !child_saturated {
                roundoff_count += 1;
            } else {
                roundoff_count = 0;
            }

            // Update totals: remove old contribution, add new
            total_estimate = total_estimate - worst.estimate + children_estimate;
            total_error = total_error - worst.error + children_error;

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

            if roundoff_count >= ROUNDOFF_STALL_LIMIT {
                roundoff_limited = true;
                break;
            }
        }

        let converged = self.tolerance_met(total_estimate, total_error);
        // A final bisection can both bring the total error within tolerance and
        // trip the roundoff counter on the same iteration (the tolerance check
        // is at the loop head, the roundoff break at its tail). `converged`
        // takes precedence, keeping the three outcomes mutually exclusive:
        // converged, roundoff-limited, or budget-exhausted.
        let roundoff_limited = roundoff_limited && !converged;

        Ok(QuadratureResult {
            value: total_estimate,
            error_estimate: total_error,
            num_evals,
            converged,
            roundoff_limited,
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
/// Uses G7-K15 with absolute and relative tolerance both set to `tol`. Inspect
/// [`converged`](QuadratureResult::converged) and
/// [`roundoff_limited`](QuadratureResult::roundoff_limited) on the result to
/// tell a met tolerance from the floating-point floor or budget exhaustion.
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
/// # Example
///
/// ```
/// use bilby::adaptive_integrate_with_breaks;
///
/// // Integrate |x| over [-1, 1] with a break at the cusp
/// let result = adaptive_integrate_with_breaks(|x: f64| x.abs(), -1.0, 1.0, &[0.0], 1e-12).unwrap();
/// assert!((result.value - 1.0).abs() < 1e-12);
/// ```
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

    /// Too many break-point subintervals for the eval budget is rejected,
    /// rather than silently overrunning max_evals during seeding.
    #[test]
    fn seed_budget_rejected_when_too_many_breaks() {
        let breaks: Vec<f64> = (1..200).map(|i| f64::from(i) / 200.0).collect();
        let r = AdaptiveIntegrator::default()
            .with_max_evals(100)
            .integrate_with_breaks(0.0, 1.0, &breaks, |x| x);
        assert!(r.is_err());
    }

    /// An integrable endpoint singularity forces refinement toward the limit of
    /// float resolution. The min-width guard drops unrefineable panels instead
    /// of spinning to max_evals, so the call terminates with a finite value.
    #[test]
    fn singular_integrand_terminates() {
        let r = AdaptiveIntegrator::default()
            .with_abs_tol(1e-13)
            .with_rel_tol(1e-13)
            .integrate(
                0.0,
                1.0,
                |x: f64| if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 },
            )
            .unwrap();
        assert!(r.value.is_finite());
        assert!((r.value - 2.0).abs() < 0.05, "value={}", r.value);
    }

    /// Reversed bounds that require subdivision must still negate the forward
    /// result. A peaked integrand forces the refinement loop (and the min-width
    /// guard) to run, where an orientation-dependent guard would wrongly drop
    /// every reversed (a > b) interval.
    #[test]
    fn reversed_bounds_needing_refinement_negate() {
        let f = |x: f64| 1.0 / (1.0 + (x - 0.3).powi(2) * 1.0e4);
        let fwd = AdaptiveIntegrator::default()
            .with_abs_tol(1e-10)
            .with_rel_tol(1e-10)
            .integrate(0.0, 1.0, f)
            .unwrap();
        let rev = AdaptiveIntegrator::default()
            .with_abs_tol(1e-10)
            .with_rel_tol(1e-10)
            .integrate(1.0, 0.0, f)
            .unwrap();
        assert!(
            fwd.converged && rev.converged,
            "fwd.conv={}, rev.conv={}",
            fwd.converged,
            rev.converged
        );
        assert!(
            (fwd.value + rev.value).abs() < 1e-9,
            "fwd={}, rev={}",
            fwd.value,
            rev.value
        );
    }

    /// Builder with higher-order pair achieves tight tolerance.
    #[test]
    fn builder_high_order_roundoff_limited() {
        // 1e-14 is tighter than the f64 roundoff floor (≈ 50·εmach·∫|sin| ≈
        // 2.2e-14 over [0,π]), so the requested tolerance cannot be certified.
        // The integrator must stop roundoff-limited (QUADPACK ier=2) — value
        // accurate to machine precision, converged=false, roundoff_limited=true
        // — rather than spinning to max_evals.
        let r = AdaptiveIntegrator::default()
            .with_pair(GKPair::G15K31)
            .with_abs_tol(1e-14)
            .with_rel_tol(1e-14)
            .with_max_evals(100_000)
            .integrate(0.0, core::f64::consts::PI, f64::sin)
            .unwrap();
        assert!((r.value - 2.0).abs() < 1e-13, "value={}", r.value);
        assert!(r.roundoff_limited, "expected roundoff_limited");
        assert!(!r.converged, "expected not converged at unachievable 1e-14");
        assert!(
            r.num_evals < 5_000,
            "did not stop early: num_evals={}",
            r.num_evals
        );
    }

    #[test]
    fn smooth_integrand_roundoff_limited_at_tight_tol() {
        // exp over [0,1] at rel_tol 1e-15 (below the floor ≈ 50·εmach·∫|exp|):
        // the value is accurate but the tolerance is unachievable.
        let r = AdaptiveIntegrator::default()
            .with_abs_tol(0.0)
            .with_rel_tol(1e-15)
            .integrate(0.0, 1.0, f64::exp)
            .unwrap();
        let exact = core::f64::consts::E - 1.0;
        assert!((r.value - exact).abs() < 1e-13, "value={}", r.value);
        assert!(r.roundoff_limited && !r.converged);
        assert!(r.num_evals < 5_000, "num_evals={}", r.num_evals);
    }

    #[test]
    fn under_resolved_oscillatory_not_falsely_converged() {
        // sin(1000x) with a budget far too small to resolve ~159 oscillations:
        // the panels saturate their resasc cap, so the roundoff guard must NOT
        // declare success on the (inaccurate) value — converged/roundoff_limited
        // may only be true if the value is in fact accurate.
        let exact = (1.0 - 1000.0_f64.cos()) / 1000.0;
        let r = AdaptiveIntegrator::default()
            .with_abs_tol(1e-12)
            .with_rel_tol(1e-12)
            .with_max_evals(2_000)
            .integrate(0.0, 1.0, |x: f64| (1000.0 * x).sin())
            .unwrap();
        assert!(
            !(r.roundoff_limited || r.converged) || (r.value - exact).abs() < 1e-6,
            "falsely reported success: value={}, exact={exact}, conv={}, ro={}",
            r.value,
            r.converged,
            r.roundoff_limited
        );
    }

    #[test]
    fn discontinuity_without_break_not_falsely_converged() {
        // A jump at x=2 with no break point. The straddling panels are
        // under-resolved (cap-saturated); success may only be reported with an
        // accurate value.
        let step = |x: f64| if x < 2.0 { 1.0 } else { 3.0 };
        let r = AdaptiveIntegrator::default()
            .with_abs_tol(1e-14)
            .with_rel_tol(1e-14)
            .integrate(0.0, 4.0, step)
            .unwrap();
        assert!(
            !(r.roundoff_limited || r.converged) || (r.value - 8.0).abs() < 1e-6,
            "value={}, conv={}, ro={}",
            r.value,
            r.converged,
            r.roundoff_limited
        );
    }

    #[test]
    fn nan_integrand_terminates_without_panic() {
        // A NaN-producing integrand makes some panel errors NaN; the heap must
        // stay well-ordered (total_cmp) and the call must return, not panic.
        let r = AdaptiveIntegrator::default().integrate(0.0, 1.0, |_x: f64| f64::NAN);
        assert!(r.is_ok());
    }

    #[test]
    fn nan_tolerance_rejected() {
        // A NaN tolerance must error, not slip past the `<= 0.0` check and spin.
        let r = AdaptiveIntegrator::default()
            .with_abs_tol(f64::NAN)
            .integrate(0.0, 1.0, |x: f64| x);
        assert!(r.is_err());
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

    /// Reversed bounds for exp(x): ∫₁⁰ exp(x) dx = -(e - 1).
    #[test]
    fn reversed_bounds_exp() {
        let forward = adaptive_integrate(f64::exp, 0.0, 1.0, 1e-12).unwrap();
        let reverse = adaptive_integrate(f64::exp, 1.0, 0.0, 1e-12).unwrap();
        assert!(
            (forward.value + reverse.value).abs() < 1e-12,
            "forward={}, reverse={}",
            forward.value,
            reverse.value
        );
    }

    /// Reversed bounds for a polynomial: ∫₁⁰ x² dx = -1/3.
    #[test]
    fn reversed_bounds_polynomial() {
        let r = adaptive_integrate(|x| x * x, 1.0, 0.0, 1e-12).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!(
            (r.value + 1.0 / 3.0).abs() < 1e-12,
            "value={}, expected -1/3",
            r.value
        );
    }

    /// Reversed bounds with break points: ∫₁⁻¹ |x| dx = -1.
    #[test]
    fn reversed_bounds_with_breaks() {
        let forward =
            adaptive_integrate_with_breaks(|x: f64| x.abs(), -1.0, 1.0, &[0.0], 1e-12).unwrap();
        let reverse =
            adaptive_integrate_with_breaks(|x: f64| x.abs(), 1.0, -1.0, &[0.0], 1e-12).unwrap();
        assert!(forward.converged, "forward err={}", forward.error_estimate);
        assert!(reverse.converged, "reverse err={}", reverse.error_estimate);
        assert!(
            (forward.value + reverse.value).abs() < 1e-12,
            "forward={}, reverse={}",
            forward.value,
            reverse.value
        );
    }

    /// Reversed bounds with break points for step function.
    #[test]
    fn reversed_bounds_step_with_breaks() {
        let step = |x: f64| if x < 2.0 { 1.0 } else { 3.0 };
        let forward = adaptive_integrate_with_breaks(step, 0.0, 4.0, &[2.0], 1e-12).unwrap();
        let reverse = adaptive_integrate_with_breaks(step, 4.0, 0.0, &[2.0], 1e-12).unwrap();
        assert!(forward.converged, "forward err={}", forward.error_estimate);
        assert!(reverse.converged, "reverse err={}", reverse.error_estimate);
        assert!(
            (forward.value + reverse.value).abs() < 1e-12,
            "forward={}, reverse={}",
            forward.value,
            reverse.value
        );
    }

    /// NaN bounds are rejected.
    #[test]
    fn nan_bounds_rejected() {
        assert!(adaptive_integrate(f64::sin, f64::NAN, 1.0, 1e-10).is_err());
        assert!(adaptive_integrate(f64::sin, 0.0, f64::NAN, 1e-10).is_err());
    }

    /// Infinite bounds are rejected.
    #[test]
    fn infinite_bounds_rejected() {
        assert!(adaptive_integrate(f64::sin, f64::INFINITY, 1.0, 1e-10).is_err());
        assert!(adaptive_integrate(f64::sin, 0.0, f64::NEG_INFINITY, 1e-10).is_err());
    }

    /// NaN break point is rejected.
    #[test]
    fn nan_break_point_rejected() {
        assert!(adaptive_integrate_with_breaks(f64::sin, 0.0, 1.0, &[f64::NAN], 1e-10).is_err());
    }

    /// max_evals smaller than one GK panel is rejected.
    #[test]
    fn max_evals_too_small() {
        let r = AdaptiveIntegrator::default()
            .with_max_evals(1)
            .integrate(0.0, 1.0, f64::sin);
        assert!(r.is_err());
    }
}
