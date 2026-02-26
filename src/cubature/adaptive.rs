//! Adaptive cubature via Genz-Malik embedded rules.
//!
//! Uses a degree-7/degree-5 rule pair for error estimation with
//! h-adaptive subdivision of the worst sub-region. The split axis
//! is chosen by the largest fourth-difference criterion.
//!
//! Practical for dimensions d ≤ ~7 (the rule uses 2^d vertex evaluations).
//! For d = 1, delegates to the 1D adaptive integrator.

use core::cmp::Ordering;

#[cfg(not(feature = "std"))]
use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;
#[cfg(feature = "std")]
use std::collections::BinaryHeap;

use crate::error::QuadratureError;
use crate::result::QuadratureResult;

/// Builder for h-adaptive cubature over a hyperrectangle.
///
/// # Example
///
/// ```
/// use bilby::cubature::adaptive_cubature;
///
/// let result = adaptive_cubature(
///     |x| (-x[0]*x[0] - x[1]*x[1]).exp(),
///     &[0.0, 0.0], &[1.0, 1.0], 1e-6,
/// ).unwrap();
/// assert!(result.is_converged());
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveCubature {
    abs_tol: f64,
    rel_tol: f64,
    max_evals: usize,
}

impl Default for AdaptiveCubature {
    fn default() -> Self {
        Self {
            abs_tol: 1.49e-8,
            rel_tol: 1.49e-8,
            max_evals: 500_000,
        }
    }
}

impl AdaptiveCubature {
    /// Set absolute tolerance.
    #[must_use]
    pub fn with_abs_tol(mut self, tol: f64) -> Self {
        self.abs_tol = tol;
        self
    }

    /// Set relative tolerance.
    #[must_use]
    pub fn with_rel_tol(mut self, tol: f64) -> Self {
        self.rel_tol = tol;
        self
    }

    /// Set maximum number of function evaluations.
    #[must_use]
    pub fn with_max_evals(mut self, n: usize) -> Self {
        self.max_evals = n;
        self
    }

    /// Adaptively integrate `f` over the hyperrectangle \[lower, upper\].
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if `lower` and `upper` have
    /// different lengths or are empty. Returns [`QuadratureError::DegenerateInterval`]
    /// if any bound is NaN.
    pub fn integrate<G>(
        &self,
        lower: &[f64],
        upper: &[f64],
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64,
    {
        let d = lower.len();
        if d == 0 || upper.len() != d {
            return Err(QuadratureError::InvalidInput(
                "lower and upper must have equal nonzero length",
            ));
        }
        for i in 0..d {
            if lower[i].is_nan() || upper[i].is_nan() {
                return Err(QuadratureError::DegenerateInterval);
            }
        }

        if d == 1 {
            // Delegate to 1D adaptive integrator
            return crate::adaptive::adaptive_integrate(
                |x: f64| f(&[x]),
                lower[0],
                upper[0],
                self.abs_tol,
            );
        }

        let mut heap: BinaryHeap<SubRegion> = BinaryHeap::new();
        let mut total_evals = 0usize;

        // Evaluate the initial region
        let detail = genz_malik_eval(d, lower, upper, &f);
        total_evals += detail.num_evals;

        heap.push(SubRegion {
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            estimate: detail.estimate,
            error: detail.error,
            split_axis: detail.split_axis,
        });

        let mut global_estimate = detail.estimate;
        let mut global_error = detail.error;

        while global_error > self.abs_tol.max(self.rel_tol * global_estimate.abs())
            && total_evals < self.max_evals
        {
            let Some(worst) = heap.pop() else { break };

            global_estimate -= worst.estimate;
            global_error -= worst.error;

            // Bisect along the split axis
            let axis = worst.split_axis;
            let mid = 0.5 * (worst.lower[axis] + worst.upper[axis]);

            let mut upper1 = worst.upper.clone();
            upper1[axis] = mid;
            let d1 = genz_malik_eval(d, &worst.lower, &upper1, &f);
            total_evals += d1.num_evals;

            let mut lower2 = worst.lower.clone();
            lower2[axis] = mid;
            let d2 = genz_malik_eval(d, &lower2, &worst.upper, &f);
            total_evals += d2.num_evals;

            global_estimate += d1.estimate + d2.estimate;
            global_error += d1.error + d2.error;

            heap.push(SubRegion {
                lower: worst.lower,
                upper: upper1,
                estimate: d1.estimate,
                error: d1.error,
                split_axis: d1.split_axis,
            });
            heap.push(SubRegion {
                lower: lower2,
                upper: worst.upper,
                estimate: d2.estimate,
                error: d2.error,
                split_axis: d2.split_axis,
            });
        }

        let converged = global_error <= self.abs_tol.max(self.rel_tol * global_estimate.abs());

        Ok(QuadratureResult {
            value: global_estimate,
            error_estimate: global_error,
            num_evals: total_evals,
            converged,
        })
    }
}

/// Convenience: adaptive cubature with default settings.
///
/// # Example
///
/// ```
/// use bilby::cubature::adaptive_cubature;
///
/// // Integral of x*y over [0,1]^2 = 1/4
/// let result = adaptive_cubature(|x| x[0] * x[1], &[0.0, 0.0], &[1.0, 1.0], 1e-10).unwrap();
/// assert!((result.value - 0.25).abs() < 1e-10);
/// ```
///
/// # Errors
///
/// Returns [`QuadratureError::InvalidInput`] if `lower` and `upper` have
/// different lengths or are empty. Returns [`QuadratureError::DegenerateInterval`]
/// if any bound is NaN.
pub fn adaptive_cubature<G>(
    f: G,
    lower: &[f64],
    upper: &[f64],
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(&[f64]) -> f64,
{
    AdaptiveCubature::default()
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .integrate(lower, upper, f)
}

/// Sub-region in the priority queue.
#[derive(Debug, Clone)]
struct SubRegion {
    lower: Vec<f64>,
    upper: Vec<f64>,
    estimate: f64,
    error: f64,
    split_axis: usize,
}

impl PartialEq for SubRegion {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl Eq for SubRegion {}

impl PartialOrd for SubRegion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SubRegion {
    fn cmp(&self, other: &Self) -> Ordering {
        self.error
            .partial_cmp(&other.error)
            .unwrap_or(Ordering::Equal)
    }
}

/// Result of a single Genz-Malik evaluation on a sub-region.
struct GMDetail {
    estimate: f64,
    error: f64,
    split_axis: usize,
    num_evals: usize,
}

/// Evaluate the Genz-Malik degree-7/degree-5 embedded rule on a sub-region.
///
/// Reference: Berntsen, Espelid, Genz (1991), "An Adaptive Algorithm for
/// Numerical Integration over an N-Dimensional Rectangular Region",
/// J. Comput. Appl. Math. 35, 179–196.
///
/// The rule evaluates f at 5 generator types:
/// 1. Center: 1 point
/// 2. Along each axis at ±λ₂: 2d points
/// 3. Along each axis at ±λ₃: 2d points
/// 4. All vertices at (±λ₄, ..., ±λ₄): 2^d points
/// 5. All pairs of axes at (±λ₅, ±λ₅, 0, ...): 2d(d-1) points
///
/// Total: 1 + 4d + 2^d + 2d(d-1) = 2^d + 2d² + 2d + 1 evaluations.
fn genz_malik_eval<G>(d: usize, lower: &[f64], upper: &[f64], f: &G) -> GMDetail
where
    G: Fn(&[f64]) -> f64,
{
    // Constants from Genz-Malik
    let lambda2 = (9.0_f64 / 70.0).sqrt();
    let lambda4 = (9.0_f64 / 10.0).sqrt();
    let lambda5 = (9.0_f64 / 19.0).sqrt();

    // Dimension is bounded by practical limits (d <= ~7), so this cast is safe.
    #[allow(clippy::cast_precision_loss)]
    let d_f = d as f64;

    // Weights for degree-7 rule (scaled for [-1,1]^d)
    let w1 = (12824.0 - 9120.0 * d_f + 400.0 * d_f * d_f) / 19683.0;
    let w2 = 980.0 / 6561.0;
    let w3 = (1820.0 - 400.0 * d_f) / 19683.0;
    let w4 = 200.0 / 19683.0;
    // d is bounded by practical limits (d <= ~7), so (1 << d) fits in i32.
    let w5 = 6859.0 / 19683.0 / f64::from(1i32 << d);

    // Weights for degree-5 rule
    let wp1 = (729.0 - 950.0 * d_f + 50.0 * d_f * d_f) / 729.0;
    let wp2 = 245.0 / 486.0;
    let wp3 = (265.0 - 100.0 * d_f) / 1458.0;
    let wp4 = 25.0 / 729.0;

    // Midpoint and half-widths
    let half_widths: Vec<f64> = (0..d).map(|j| 0.5 * (upper[j] - lower[j])).collect();
    let midpoints: Vec<f64> = (0..d).map(|j| 0.5 * (lower[j] + upper[j])).collect();
    let volume: f64 = half_widths.iter().map(|h| 2.0 * h).product();

    let mut x = midpoints.clone();
    let mut num_evals = 0;

    // Type 1: center
    let f_center = f(&x);
    num_evals += 1;

    // Type 2 & 3: along each axis at ±λ₂ and ±λ₃
    let mut sum_2 = 0.0;
    let mut sum_3 = 0.0;
    let mut fourth_diffs = vec![0.0; d]; // for split axis selection

    for i in 0..d {
        x[i] = midpoints[i] + lambda2 * half_widths[i];
        let f_plus2 = f(&x);
        x[i] = midpoints[i] - lambda2 * half_widths[i];
        let f_minus2 = f(&x);
        sum_2 += f_plus2 + f_minus2;

        x[i] = midpoints[i] + lambda4 * half_widths[i];
        let f_plus4 = f(&x);
        x[i] = midpoints[i] - lambda4 * half_widths[i];
        let f_minus4 = f(&x);
        sum_3 += f_plus4 + f_minus4;

        // Fourth difference for split axis selection
        fourth_diffs[i] = (f_plus2 + f_minus2 - 2.0 * f_center).abs()
            - (f_plus4 + f_minus4 - 2.0 * f_center).abs();

        x[i] = midpoints[i]; // restore
        num_evals += 4;
    }

    // Generator 5: all vertices at (±λ₅, ..., ±λ₅)  — 2^d points
    let mut sum_vertices = 0.0;
    let n_vertices = 1usize << d;
    for bits in 0..n_vertices {
        for j in 0..d {
            let sign = if (bits >> j) & 1 == 0 { 1.0 } else { -1.0 };
            x[j] = midpoints[j] + sign * lambda5 * half_widths[j];
        }
        sum_vertices += f(&x);
        num_evals += 1;
    }
    // Restore center
    x.copy_from_slice(&midpoints);

    // Generator 4: all pairs of axes at (±λ₄, ±λ₄, 0, ...) — 2d(d-1) points
    let mut sum_faces = 0.0;
    for i in 0..d {
        for j in (i + 1)..d {
            for si in [-1.0_f64, 1.0] {
                for sj in [-1.0_f64, 1.0] {
                    x[i] = midpoints[i] + si * lambda4 * half_widths[i];
                    x[j] = midpoints[j] + sj * lambda4 * half_widths[j];
                    sum_faces += f(&x);
                    num_evals += 1;
                    x[i] = midpoints[i];
                    x[j] = midpoints[j];
                }
            }
        }
    }

    // Genz-Malik weights are normalised so that w_sum = 1 for constant f=1.
    // The integral estimate is therefore volume * weighted_sum.
    let jac = volume;

    // Degree-7 estimate
    let est7 = jac * (w1 * f_center + w2 * sum_2 + w3 * sum_3 + w4 * sum_faces + w5 * sum_vertices);

    // Degree-5 estimate (no vertex term)
    let est5 = jac * (wp1 * f_center + wp2 * sum_2 + wp3 * sum_3 + wp4 * sum_faces);

    let error = (est7 - est5).abs();

    // Choose split axis: largest fourth difference
    let split_axis = fourth_diffs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map_or(0, |(i, _)| i);

    GMDetail {
        estimate: est7,
        error,
        split_axis,
        num_evals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_input() {
        assert!(adaptive_cubature(|_| 1.0, &[], &[], 1e-8).is_err());
        assert!(adaptive_cubature(|_| 1.0, &[0.0], &[1.0, 2.0], 1e-8).is_err());
    }

    /// Constant function over [0,1]^2.
    #[test]
    fn constant_2d() {
        let result = adaptive_cubature(|_| 5.0, &[0.0, 0.0], &[1.0, 1.0], 1e-10).unwrap();
        assert!((result.value - 5.0).abs() < 1e-10, "value={}", result.value);
        assert!(result.is_converged());
    }

    /// Polynomial: integral of x*y over [0,1]^2 = 1/4.
    #[test]
    fn polynomial_2d() {
        let result = adaptive_cubature(|x| x[0] * x[1], &[0.0, 0.0], &[1.0, 1.0], 1e-10).unwrap();
        assert!(
            (result.value - 0.25).abs() < 1e-10,
            "value={}",
            result.value
        );
    }

    /// Gaussian in 2D.
    #[test]
    fn gaussian_2d() {
        let result = adaptive_cubature(
            |x| (-x[0] * x[0] - x[1] * x[1]).exp(),
            &[0.0, 0.0],
            &[1.0, 1.0],
            1e-6,
        )
        .unwrap();
        // (erf(1) * sqrt(pi)/2)^2 ≈ 0.557...
        let one_d = 0.7468241328124271;
        let expected = one_d * one_d;
        assert!(
            (result.value - expected).abs() < 1e-6,
            "value={}, expected={expected}",
            result.value
        );
        assert!(result.is_converged());
    }

    /// 3D integral.
    #[test]
    fn separable_3d() {
        let result = adaptive_cubature(
            |x| x[0] * x[0] + x[1] * x[1] + x[2] * x[2],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            1e-10,
        )
        .unwrap();
        // 3 * (1/3) = 1
        assert!((result.value - 1.0).abs() < 1e-10, "value={}", result.value);
    }

    /// 1D delegation.
    #[test]
    fn delegates_to_1d() {
        let result =
            adaptive_cubature(|x| x[0].sin(), &[0.0], &[core::f64::consts::PI], 1e-10).unwrap();
        assert!((result.value - 2.0).abs() < 1e-10, "value={}", result.value);
    }

    /// Non-convergence with tight budget.
    #[test]
    fn non_convergence() {
        let result = AdaptiveCubature::default()
            .with_abs_tol(1e-15)
            .with_rel_tol(1e-15)
            .with_max_evals(100)
            .integrate(&[0.0, 0.0], &[1.0, 1.0], |x| {
                (100.0 * x[0]).sin() * (100.0 * x[1]).cos()
            })
            .unwrap();
        assert!(!result.is_converged());
        assert!(result.value.is_finite());
    }
}
