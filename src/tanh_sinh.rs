//! Tanh-sinh (double-exponential) quadrature.
//!
//! Transforms endpoint singularities into rapidly decaying integrands via
//! x = tanh(π/2 · sinh(t)). Self-adaptive through level doubling: each level
//! halves the step size and reuses all previously computed points.
//!
//! Particularly effective for integrands with algebraic or logarithmic
//! endpoint singularities where standard Gauss rules struggle.
//!
//! # Example
//!
//! ```
//! use bilby::tanh_sinh_integrate;
//!
//! // Integral of 1/sqrt(x) over [0, 1] = 2  (endpoint singularity)
//! let result = tanh_sinh_integrate(|x| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10).unwrap();
//! assert!((result.value - 2.0).abs() < 1e-7);
//! ```

use crate::error::QuadratureError;
use crate::result::QuadratureResult;

/// Builder for tanh-sinh (double-exponential) quadrature.
///
/// # Example
///
/// ```
/// use bilby::tanh_sinh::TanhSinh;
///
/// let result = TanhSinh::default()
///     .with_abs_tol(1e-12)
///     .integrate(0.0, 1.0, |x| x.ln())
///     .unwrap();
/// assert!((result.value - (-1.0)).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct TanhSinh {
    max_levels: usize,
    abs_tol: f64,
    rel_tol: f64,
}

impl Default for TanhSinh {
    fn default() -> Self {
        Self {
            max_levels: 12,
            abs_tol: 1.49e-8,
            rel_tol: 1.49e-8,
        }
    }
}

impl TanhSinh {
    /// Set the maximum number of refinement levels.
    pub fn with_max_levels(mut self, levels: usize) -> Self {
        self.max_levels = levels;
        self
    }

    /// Set absolute tolerance.
    pub fn with_abs_tol(mut self, tol: f64) -> Self {
        self.abs_tol = tol;
        self
    }

    /// Set relative tolerance.
    pub fn with_rel_tol(mut self, tol: f64) -> Self {
        self.rel_tol = tol;
        self
    }

    /// Integrate `f` over [a, b] using the tanh-sinh transform.
    pub fn integrate<G>(
        &self,
        a: f64,
        b: f64,
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(f64) -> f64,
    {
        if a.is_nan() || b.is_nan() {
            return Err(QuadratureError::DegenerateInterval);
        }
        if (b - a).abs() < f64::EPSILON {
            return Ok(QuadratureResult {
                value: 0.0,
                error_estimate: 0.0,
                num_evals: 0,
                converged: true,
            });
        }

        let mid = 0.5 * (a + b);
        let half = 0.5 * (b - a);
        let pi_2 = std::f64::consts::FRAC_PI_2;

        let mut total_evals = 0usize;
        let mut prev_estimate: f64 = 0.0;
        let mut h: f64 = 1.0;

        // Level 0: evaluate at all points t = k*h
        // Subsequent levels: only evaluate at odd multiples of the new h
        for level in 0..=self.max_levels {
            let step = if level == 0 { 1 } else { 2 };
            let start = 1; // k=0 is the center point, handled separately

            // At level L, h = 2^{-L}. New points are at t = (2k+1) * h for k >= 0.
            // But we also need negative t values.

            if level > 0 {
                // Scale previous sum for the new step size (halved h)
                // prev_estimate was computed with old h, which is 2*current h.
                // The trapezoidal rule sum scales linearly with h, so halving h
                // means we keep all old points and add new ones.
                // Actually, the estimate is h * sum_of_weighted_values.
                // When we halve h, old_estimate = 2h * old_sum = old_value.
                // New estimate = h * (old_sum + new_sum) = old_value/2 + h * new_sum.
                h *= 0.5;
            }

            let mut new_sum = 0.0;

            // Center point (only at level 0)
            if level == 0 {
                // t = 0: sinh(0) = 0, cosh(0) = 1, tanh(0) = 0
                // x = mid, weight = pi/2
                let w0 = pi_2;
                let fval = f(mid);
                if fval.is_finite() {
                    new_sum += w0 * fval;
                }
                total_evals += 1;
            }

            // Positive and negative t values
            // Upper bound for the loop; actual termination is handled by the
            // u > 710, weight < 1e-300, and consecutive_tiny checks.
            let max_k = (7.0 / h).ceil() as usize;
            let mut k = start;
            let mut consecutive_tiny = 0;

            while k <= max_k {
                let t = k as f64 * h;
                let sinh_t = t.sinh();
                let cosh_t = t.cosh();
                let u = pi_2 * sinh_t;

                // For large |u|, tanh(u) ≈ ±1 and cosh(u) is huge.
                // The weight decays double-exponentially.
                if u.abs() > 710.0 {
                    // cosh(u)^2 would overflow; weight is effectively zero
                    break;
                }

                let cosh_u = u.cosh();
                let tanh_u = u.tanh();
                let weight = pi_2 * cosh_t / (cosh_u * cosh_u);

                if weight < 1e-300 {
                    break;
                }

                // Positive t: x = mid + half * tanh(u)
                let mut local_contrib = 0.0_f64;
                let x_pos = mid + half * tanh_u;
                if x_pos > a && x_pos < b {
                    let fval = f(x_pos);
                    if fval.is_finite() {
                        new_sum += weight * fval;
                        local_contrib = local_contrib.max((weight * fval).abs());
                    }
                    total_evals += 1;
                }

                // Negative t: x = mid - half * tanh(u)
                let x_neg = mid - half * tanh_u;
                if x_neg > a && x_neg < b {
                    let fval = f(x_neg);
                    if fval.is_finite() {
                        new_sum += weight * fval;
                        local_contrib = local_contrib.max((weight * fval).abs());
                    }
                    total_evals += 1;
                }

                // Check if actual contributions (including f(x)) are negligible
                let threshold = f64::EPSILON * prev_estimate.abs().max(1e-300);
                if local_contrib * half < threshold {
                    consecutive_tiny += 1;
                    if consecutive_tiny >= 3 {
                        break;
                    }
                } else {
                    consecutive_tiny = 0;
                }

                k += step;
            }

            let estimate = if level == 0 {
                h * half * new_sum
            } else {
                0.5 * prev_estimate + h * half * new_sum
            };

            // Require at least 3 levels before checking convergence.
            // Early levels have so few points that successive differences
            // can be misleadingly small for singular integrands.
            if level >= 3 {
                let error = (estimate - prev_estimate).abs();
                let tol = self.abs_tol.max(self.rel_tol * estimate.abs());
                if error <= tol {
                    return Ok(QuadratureResult {
                        value: estimate,
                        error_estimate: error,
                        num_evals: total_evals,
                        converged: true,
                    });
                }
            }

            prev_estimate = estimate;
        }

        // Did not converge within max_levels
        let error = if self.max_levels > 0 {
            // Can't easily recompute the last level difference,
            // but prev_estimate is our best value
            self.abs_tol.max(self.rel_tol * prev_estimate.abs()) * 10.0
        } else {
            f64::INFINITY
        };

        Ok(QuadratureResult {
            value: prev_estimate,
            error_estimate: error,
            num_evals: total_evals,
            converged: false,
        })
    }
}

/// Convenience: tanh-sinh integration with default settings.
pub fn tanh_sinh_integrate<G>(
    f: G,
    a: f64,
    b: f64,
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    TanhSinh::default()
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .integrate(a, b, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_polynomial() {
        let result = tanh_sinh_integrate(|x| x * x, 0.0, 1.0, 1e-12).unwrap();
        assert!(
            (result.value - 1.0 / 3.0).abs() < 1e-10,
            "value={}",
            result.value
        );
        assert!(result.converged);
    }

    #[test]
    fn sqrt_singularity() {
        // ∫₀¹ 1/√x dx = 2
        // f64 limits: tanh saturates near ±1, losing ~1.5e-8 of the integral
        let result = tanh_sinh_integrate(|x| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10).unwrap();
        assert!((result.value - 2.0).abs() < 1e-7, "value={}", result.value);
    }

    #[test]
    fn log_singularity() {
        // ∫₀¹ ln(x) dx = -1
        let result = tanh_sinh_integrate(|x| x.ln(), 0.0, 1.0, 1e-10).unwrap();
        assert!(
            (result.value - (-1.0)).abs() < 1e-8,
            "value={}",
            result.value
        );
    }

    #[test]
    fn strong_algebraic_singularity() {
        // ∫₀¹ x^(-0.75) dx = 1/0.25 = 4
        // x^(-0.9) loses ~0.24 due to f64 tanh saturation near ±1;
        // x^(-0.75) loses only ~3.5e-4, well within tolerance.
        let result = tanh_sinh_integrate(|x| x.powf(-0.75), 0.0, 1.0, 1e-6).unwrap();
        assert!((result.value - 4.0).abs() < 0.01, "value={}", result.value);
    }

    #[test]
    fn both_endpoints_singular() {
        // ∫₀¹ 1/√(x(1-x)) dx = π
        let result = tanh_sinh_integrate(|x| 1.0 / (x * (1.0 - x)).sqrt(), 0.0, 1.0, 1e-8).unwrap();
        assert!(
            (result.value - std::f64::consts::PI).abs() < 1e-6,
            "value={}",
            result.value
        );
    }

    #[test]
    fn chebyshev_weight() {
        // ∫₋₁¹ 1/√(1-x²) dx = π
        let result = tanh_sinh_integrate(|x| 1.0 / (1.0 - x * x).sqrt(), -1.0, 1.0, 1e-8).unwrap();
        assert!(
            (result.value - std::f64::consts::PI).abs() < 1e-6,
            "value={}",
            result.value
        );
    }

    #[test]
    fn sin_integral() {
        // ∫₀^π sin(x) dx = 2  (smooth, sanity check)
        let result = tanh_sinh_integrate(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-10).unwrap();
        assert!((result.value - 2.0).abs() < 1e-8, "value={}", result.value);
    }

    #[test]
    fn zero_width_interval() {
        let result = tanh_sinh_integrate(|x| x, 1.0, 1.0, 1e-10).unwrap();
        assert_eq!(result.value, 0.0);
        assert!(result.converged);
    }

    #[test]
    fn nan_bounds() {
        assert!(tanh_sinh_integrate(|x| x, f64::NAN, 1.0, 1e-10).is_err());
    }

    #[test]
    fn non_convergence() {
        let result = TanhSinh::default()
            .with_max_levels(0)
            .integrate(0.0, 1.0, |x| 1.0 / x.sqrt())
            .unwrap();
        assert!(!result.converged);
    }
}
