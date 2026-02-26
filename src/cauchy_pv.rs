//! Cauchy principal value integration.
//!
//! Computes PV ∫ₐᵇ f(x)/(x-c) dx where c ∈ (a, b) using the subtraction
//! technique: rewrite as a regular integral plus an analytic logarithmic term.
//!
//! # Example
//!
//! ```
//! use bilby::cauchy_pv::pv_integrate;
//!
//! // PV ∫₀¹ x²/(x - 0.3) dx = 0.8 + 0.09 * ln(7/3)
//! let exact = 0.8 + 0.09 * (7.0_f64 / 3.0).ln();
//! let result = pv_integrate(|x| x * x, 0.0, 1.0, 0.3, 1e-10).unwrap();
//! assert!((result.value - exact).abs() < 1e-8);
//! ```

use crate::adaptive::AdaptiveIntegrator;
use crate::error::QuadratureError;
use crate::result::QuadratureResult;

#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Builder for Cauchy principal value integration.
///
/// Computes PV ∫ₐᵇ f(x)/(x-c) dx via the subtraction technique:
///
/// PV ∫ₐᵇ f(x)/(x-c) dx = ∫ₐᵇ \[f(x) - f(c)\]/(x-c) dx + f(c) · ln((b-c)/(c-a))
///
/// The first integral has a removable singularity at x = c and is computed
/// using adaptive quadrature with a break point at c.
#[derive(Debug, Clone)]
pub struct CauchyPV {
    abs_tol: f64,
    rel_tol: f64,
    max_evals: usize,
}

impl Default for CauchyPV {
    fn default() -> Self {
        Self {
            abs_tol: 1.49e-8,
            rel_tol: 1.49e-8,
            max_evals: 500_000,
        }
    }
}

impl CauchyPV {
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

    /// Compute PV ∫ₐᵇ f(x)/(x-c) dx.
    ///
    /// The singularity c must be strictly inside (a, b).
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::DegenerateInterval`] if any bound or `c` is NaN.
    /// Returns [`QuadratureError::InvalidInput`] if `c` is not strictly inside
    /// `(a, b)` or if `f(c)` is not finite.
    #[allow(clippy::many_single_char_names)] // a, b, c, f, g are conventional in quadrature
    pub fn integrate<G>(
        &self,
        a: f64,
        b: f64,
        c: f64,
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(f64) -> f64,
    {
        // Validate inputs
        if a.is_nan() || b.is_nan() || c.is_nan() {
            return Err(QuadratureError::DegenerateInterval);
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if c <= lo || c >= hi {
            return Err(QuadratureError::InvalidInput(
                "singularity c must be strictly inside (a, b)",
            ));
        }

        // Evaluate f at the singularity
        let fc = f(c);
        if !fc.is_finite() {
            return Err(QuadratureError::InvalidInput(
                "f(c) must be finite for the subtraction technique",
            ));
        }

        // Analytic term: f(c) * ln((b - c) / (c - a))
        let log_term = fc * ((b - c) / (c - a)).ln();

        // The subtracted integrand: g(x) = [f(x) - f(c)] / (x - c)
        // This has a removable singularity at x = c.
        let guard = 1e-15 * (b - a);
        let g = |x: f64| -> f64 {
            if (x - c).abs() < guard {
                0.0
            } else {
                (f(x) - fc) / (x - c)
            }
        };

        // Integrate g over [a, b] with a break point at c
        let adaptive_result = AdaptiveIntegrator::default()
            .with_abs_tol(self.abs_tol)
            .with_rel_tol(self.rel_tol)
            .with_max_evals(self.max_evals)
            .integrate_with_breaks(a, b, &[c], g)?;

        Ok(QuadratureResult {
            value: adaptive_result.value + log_term,
            error_estimate: adaptive_result.error_estimate,
            num_evals: adaptive_result.num_evals + 1, // +1 for f(c)
            converged: adaptive_result.converged,
        })
    }
}

/// Convenience: Cauchy principal value integration with default settings.
///
/// # Example
///
/// ```
/// use bilby::pv_integrate;
///
/// // PV integral of x^2/(x - 0.3) over [0, 1]
/// let exact = 0.8 + 0.09 * (7.0_f64 / 3.0).ln();
/// let result = pv_integrate(|x| x * x, 0.0, 1.0, 0.3, 1e-10).unwrap();
/// assert!((result.value - exact).abs() < 1e-7);
/// ```
///
/// # Errors
///
/// Returns [`QuadratureError::DegenerateInterval`] if any bound or `c` is NaN.
/// Returns [`QuadratureError::InvalidInput`] if `c` is not strictly inside
/// `(a, b)` or if `f(c)` is not finite.
pub fn pv_integrate<G>(
    f: G,
    a: f64,
    b: f64,
    c: f64,
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    CauchyPV::default()
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .integrate(a, b, c, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_constant() {
        // PV ∫₀¹ 1/(x - 0.5) dx = ln(0.5/0.5) = 0
        let result = pv_integrate(|_| 1.0, 0.0, 1.0, 0.5, 1e-10).unwrap();
        assert!(
            result.value.abs() < 1e-8,
            "value={}, expected 0",
            result.value
        );
    }

    #[test]
    fn odd_integrand() {
        // PV ∫₋₁¹ 1/x dx = 0  (odd function)
        let result = pv_integrate(|_| 1.0, -1.0, 1.0, 0.0, 1e-10).unwrap();
        assert!(
            result.value.abs() < 1e-8,
            "value={}, expected 0",
            result.value
        );
    }

    #[test]
    fn polynomial_f() {
        // PV ∫₀¹ x²/(x - 0.3) dx
        // = ∫₀¹ (x + 0.3) dx + 0.09 * ln(0.7/0.3)
        // = [x²/2 + 0.3x]₀¹ + 0.09 * ln(7/3)
        // = 0.8 + 0.09 * ln(7/3)
        let exact = 0.8 + 0.09 * (7.0_f64 / 3.0).ln();
        let result = pv_integrate(|x| x * x, 0.0, 1.0, 0.3, 1e-10).unwrap();
        assert!(
            (result.value - exact).abs() < 1e-7,
            "value={}, expected={exact}",
            result.value
        );
    }

    #[test]
    fn linear_f() {
        // PV ∫₀² x/(x - 1) dx
        // = ∫₀² [x - 1]/(x - 1) dx + 1 * ln(1/1)
        // = ∫₀² 1 dx + 0 = 2
        let result = pv_integrate(|x| x, 0.0, 2.0, 1.0, 1e-10).unwrap();
        assert!(
            (result.value - 2.0).abs() < 1e-7,
            "value={}, expected=2",
            result.value
        );
    }

    #[test]
    fn c_outside_interval() {
        assert!(pv_integrate(|x| x, 0.0, 1.0, 2.0, 1e-10).is_err());
    }

    #[test]
    fn c_at_endpoint() {
        assert!(pv_integrate(|x| x, 0.0, 1.0, 0.0, 1e-10).is_err());
        assert!(pv_integrate(|x| x, 0.0, 1.0, 1.0, 1e-10).is_err());
    }

    #[test]
    fn nan_inputs() {
        assert!(pv_integrate(|x| x, f64::NAN, 1.0, 0.5, 1e-10).is_err());
        assert!(pv_integrate(|x| x, 0.0, 1.0, f64::NAN, 1e-10).is_err());
    }
}
