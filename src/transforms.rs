//! Domain transforms for semi-infinite and infinite intervals.
//!
//! Maps infinite integration domains to finite intervals via algebraic
//! substitutions, then delegates to adaptive quadrature on the finite domain.
//!
//! # Transforms
//!
//! | Domain | Substitution | Mapped interval |
//! |--------|-------------|-----------------|
//! | \[a, ∞) | x = a + t/(1-t) | \[0, 1\] |
//! | (-∞, b\] | x = b - t/(1-t) | \[0, 1\] |
//! | (-∞, ∞) | x = t/(1-t²) | (-1, 1) |
//!
//! # Example
//!
//! ```
//! use bilby::integrate_semi_infinite_upper;
//!
//! // Integral of exp(-x) over [0, inf) = 1
//! let r = integrate_semi_infinite_upper(|x: f64| (-x).exp(), 0.0, 1e-10).unwrap();
//! assert!((r.value - 1.0).abs() < 1e-8);
//! ```

use crate::adaptive::AdaptiveIntegrator;
use crate::error::QuadratureError;
use crate::gauss_kronrod::GKPair;
use crate::result::QuadratureResult;

/// Integrate `f` over \[a, ∞) using adaptive quadrature with a domain transform.
///
/// Uses the substitution x = a + t/(1-t), mapping \[a, ∞) to \[0, 1\].
/// The Jacobian factor is 1/(1-t)².
///
/// Best suited for integrands that decay at least as fast as 1/x² at infinity.
/// For slower decay, consider splitting the integral at a finite cutoff.
///
/// # Errors
///
/// Returns [`QuadratureError::InvalidInput`] if `a` is not finite or `tol` is not positive.
///
/// # Example
///
/// ```
/// use bilby::integrate_semi_infinite_upper;
///
/// // Integral of exp(-x) over [0, inf) = 1
/// let r = integrate_semi_infinite_upper(|x: f64| (-x).exp(), 0.0, 1e-10).unwrap();
/// assert!((r.value - 1.0).abs() < 1e-8);
/// ```
pub fn integrate_semi_infinite_upper<G>(
    f: G,
    a: f64,
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    if !a.is_finite() {
        return Err(QuadratureError::InvalidInput(
            "lower bound must be finite for semi-infinite integration",
        ));
    }

    // x = a + t/(1-t), dx = 1/(1-t)^2 dt, t in [0, 1)
    let transformed = move |t: f64| {
        if t >= 1.0 {
            return 0.0;
        }
        let one_minus_t = 1.0 - t;
        let x = a + t / one_minus_t;
        let jacobian = 1.0 / (one_minus_t * one_minus_t);
        // Jacobian overflow just inside the endpoint can give 0*inf = NaN or
        // finite*inf = inf; treat a non-finite contribution as a zero tail (the
        // QUADPACK qagi guard) so one bad panel cannot poison the estimate. Note
        // this also masks a genuine non-finite f at an interior mapped point.
        let v = f(x) * jacobian;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    };

    AdaptiveIntegrator::default()
        .with_pair(GKPair::G10K21)
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .with_max_evals(50_000)
        .integrate(0.0, 1.0, transformed)
}

/// Integrate `f` over (-∞, b\] using adaptive quadrature with a domain transform.
///
/// Uses the substitution x = b - t/(1-t), mapping (-∞, b\] to \[0, 1\].
///
/// # Errors
///
/// Returns [`QuadratureError::InvalidInput`] if `b` is not finite or `tol` is not positive.
///
/// # Example
///
/// ```
/// use bilby::integrate_semi_infinite_lower;
///
/// // Integral of exp(x) over (-inf, 0] = 1
/// let r = integrate_semi_infinite_lower(|x: f64| x.exp(), 0.0, 1e-10).unwrap();
/// assert!((r.value - 1.0).abs() < 1e-8);
/// ```
pub fn integrate_semi_infinite_lower<G>(
    f: G,
    b: f64,
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    if !b.is_finite() {
        return Err(QuadratureError::InvalidInput(
            "upper bound must be finite for semi-infinite integration",
        ));
    }

    // x = b - t/(1-t), dx = -1/(1-t)^2 dt, t in [0, 1)
    // The negative sign is absorbed: integral from -inf to b = integral from 0 to 1
    // of f(b - t/(1-t)) / (1-t)^2 dt
    let transformed = move |t: f64| {
        if t >= 1.0 {
            return 0.0;
        }
        let one_minus_t = 1.0 - t;
        let x = b - t / one_minus_t;
        let jacobian = 1.0 / (one_minus_t * one_minus_t);
        // Jacobian overflow just inside the endpoint can give 0*inf = NaN or
        // finite*inf = inf; treat a non-finite contribution as a zero tail (the
        // QUADPACK qagi guard) so one bad panel cannot poison the estimate. Note
        // this also masks a genuine non-finite f at an interior mapped point.
        let v = f(x) * jacobian;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    };

    AdaptiveIntegrator::default()
        .with_pair(GKPair::G10K21)
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .with_max_evals(50_000)
        .integrate(0.0, 1.0, transformed)
}

/// Integrate `f` over (-∞, ∞) using adaptive quadrature with a domain transform.
///
/// Uses the substitution x = t/(1-t²), mapping (-∞, ∞) to (-1, 1).
/// The Jacobian factor is (1+t²)/(1-t²)².
///
/// # Errors
///
/// Returns [`QuadratureError::InvalidInput`] if `tol` is not positive.
///
/// # Example
///
/// ```
/// use bilby::integrate_infinite;
///
/// // Integral of exp(-x^2) over (-inf, inf) = sqrt(pi)
/// let r = integrate_infinite(|x: f64| (-x * x).exp(), 1e-10).unwrap();
/// assert!((r.value - core::f64::consts::PI.sqrt()).abs() < 1e-8);
/// ```
pub fn integrate_infinite<G>(f: G, tol: f64) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    // x = t/(1-t^2), dx = (1+t^2)/(1-t^2)^2 dt, t in (-1, 1)
    let transformed = move |t: f64| {
        let t2 = t * t;
        if t2 >= 1.0 {
            return 0.0;
        }
        let one_minus_t2 = 1.0 - t2;
        let x = t / one_minus_t2;
        let jacobian = (1.0 + t2) / (one_minus_t2 * one_minus_t2);
        // Jacobian overflow just inside the endpoint can give 0*inf = NaN or
        // finite*inf = inf; treat a non-finite contribution as a zero tail (the
        // QUADPACK qagi guard) so one bad panel cannot poison the estimate. Note
        // this also masks a genuine non-finite f at an interior mapped point.
        let v = f(x) * jacobian;
        if v.is_finite() {
            v
        } else {
            0.0
        }
    };

    AdaptiveIntegrator::default()
        .with_pair(GKPair::G10K21)
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .with_max_evals(50_000)
        .integrate(-1.0, 1.0, transformed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// A non-finite integrand value at a reachable mapped node must not poison
    /// the whole estimate. f is +inf beyond |x| = 50, which the outer nodes of
    /// the very first panel already reach (|x| up to ~113); the finiteness guard
    /// masks that contribution to 0 so the result stays finite instead of NaN.
    #[test]
    fn non_finite_integrand_contribution_does_not_poison() {
        let f = |x: f64| if x.abs() > 50.0 { f64::INFINITY } else { 1.0 };
        let r = integrate_infinite(f, 1e-8).unwrap();
        assert!(r.value.is_finite(), "value={}", r.value);
    }

    /// exp(-x) over [0, inf) = 1.
    #[test]
    fn exp_decay_upper() {
        let r = integrate_semi_infinite_upper(|x: f64| (-x).exp(), 0.0, 1e-10).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!((r.value - 1.0).abs() < 1e-8, "value={}", r.value);
    }

    /// exp(-x^2) over [0, inf) = sqrt(pi)/2.
    #[test]
    fn gaussian_half_line() {
        let expected = PI.sqrt() / 2.0;
        let r = integrate_semi_infinite_upper(|x: f64| (-x * x).exp(), 0.0, 1e-10).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!(
            (r.value - expected).abs() < 1e-8,
            "value={}, expected={}",
            r.value,
            expected
        );
    }

    /// 1/(1+x^2) over [0, inf) = pi/2.
    #[test]
    fn lorentzian_upper() {
        let r = integrate_semi_infinite_upper(|x: f64| 1.0 / (1.0 + x * x), 0.0, 1e-10).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!(
            (r.value - PI / 2.0).abs() < 1e-8,
            "value={}, expected={}",
            r.value,
            PI / 2.0
        );
    }

    /// exp(x) over (-inf, 0] = 1.
    #[test]
    fn exp_growth_lower() {
        let r = integrate_semi_infinite_lower(f64::exp, 0.0, 1e-10).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!((r.value - 1.0).abs() < 1e-8, "value={}", r.value);
    }

    /// exp(-x^2) over (-inf, inf) = sqrt(pi).
    #[test]
    fn gaussian_full_line() {
        let expected = PI.sqrt();
        let r = integrate_infinite(|x: f64| (-x * x).exp(), 1e-10).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!(
            (r.value - expected).abs() < 1e-8,
            "value={}, expected={}",
            r.value,
            expected
        );
    }

    /// 1/(1+x^2) over (-inf, inf) = pi.
    #[test]
    fn lorentzian_full_line() {
        let r = integrate_infinite(|x: f64| 1.0 / (1.0 + x * x), 1e-10).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!(
            (r.value - PI).abs() < 1e-8,
            "value={}, expected={}",
            r.value,
            PI
        );
    }

    /// Non-finite bound is rejected.
    #[test]
    fn non_finite_bound() {
        let r = integrate_semi_infinite_upper(f64::exp, f64::INFINITY, 1e-10);
        assert!(r.is_err());
        let r = integrate_semi_infinite_lower(f64::exp, f64::NEG_INFINITY, 1e-10);
        assert!(r.is_err());
    }

    /// Shifted semi-infinite: exp(-x) over [2, inf) = exp(-2).
    #[test]
    fn shifted_semi_infinite() {
        let expected = (-2.0_f64).exp();
        let r = integrate_semi_infinite_upper(|x: f64| (-x).exp(), 2.0, 1e-10).unwrap();
        assert!(r.converged, "err={}", r.error_estimate);
        assert!(
            (r.value - expected).abs() < 1e-8,
            "value={}, expected={}",
            r.value,
            expected
        );
    }
}
