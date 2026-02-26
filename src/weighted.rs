//! Weighted integration: ∫ f(x) · w(x) dx for known weight functions.
//!
//! Dispatches to the appropriate Gaussian quadrature rule for each weight
//! function family, providing a unified API for product integration.
//!
//! # Example
//!
//! ```
//! use bilby::weighted::{weighted_integrate, WeightFunction};
//!
//! // ∫₋₁¹ 1/√(1-x²) dx = π  (Chebyshev Type I weight)
//! let result = weighted_integrate(|_| 1.0, WeightFunction::ChebyshevI, 20).unwrap();
//! assert!((result - core::f64::consts::PI).abs() < 1e-12);
//! ```

use crate::error::QuadratureError;
use crate::gauss_chebyshev::{GaussChebyshevFirstKind, GaussChebyshevSecondKind};
use crate::gauss_hermite::GaussHermite;
use crate::gauss_jacobi::GaussJacobi;
use crate::gauss_laguerre::GaussLaguerre;

#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Known weight functions for product integration.
#[derive(Debug, Clone)]
pub enum WeightFunction {
    /// w(x) = (1-x)^α · (1+x)^β on \[-1, 1\]
    Jacobi { alpha: f64, beta: f64 },
    /// w(x) = x^α · e^(-x) on \[0, ∞)
    Laguerre { alpha: f64 },
    /// w(x) = e^(-x²) on (-∞, ∞)
    Hermite,
    /// w(x) = 1/√(1 - x²) on (-1, 1)
    ChebyshevI,
    /// w(x) = √(1 - x²) on \[-1, 1\]
    ChebyshevII,
    /// w(x) = -ln(x) on (0, 1\]
    LogWeight,
}

/// Builder for weighted integration.
///
/// # Example
///
/// ```
/// use bilby::weighted::{WeightedIntegrator, WeightFunction};
///
/// let wi = WeightedIntegrator::new(WeightFunction::Hermite, 20).unwrap();
/// // ∫₋∞^∞ 1 · e^(-x²) dx = √π
/// let result = wi.integrate(|_| 1.0);
/// assert!((result - core::f64::consts::PI.sqrt()).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct WeightedIntegrator {
    weight: WeightFunction,
    order: usize,
}

impl WeightedIntegrator {
    /// Create a new weighted integrator.
    pub fn new(weight: WeightFunction, order: usize) -> Result<Self, QuadratureError> {
        if order == 0 {
            return Err(QuadratureError::ZeroOrder);
        }
        Ok(Self { weight, order })
    }

    /// Set the quadrature order.
    pub fn with_order(mut self, n: usize) -> Self {
        self.order = n;
        self
    }

    /// Integrate f(x) · w(x) over the natural domain of w.
    ///
    /// - Jacobi: \[-1, 1\]
    /// - Laguerre: \[0, ∞)
    /// - Hermite: (-∞, ∞)
    /// - ChebyshevI/II: \[-1, 1\]
    /// - LogWeight: (0, 1\]
    pub fn integrate<G>(&self, f: G) -> f64
    where
        G: Fn(f64) -> f64,
    {
        match &self.weight {
            WeightFunction::Jacobi { alpha, beta } => {
                let gj = GaussJacobi::new(self.order, *alpha, *beta).unwrap();
                let rule = gj.rule();
                let mut sum = 0.0;
                for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
                    sum += weight * f(*node);
                }
                sum
            }
            WeightFunction::Laguerre { alpha } => {
                let gl = GaussLaguerre::new(self.order, *alpha).unwrap();
                let rule = gl.rule();
                let mut sum = 0.0;
                for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
                    sum += weight * f(*node);
                }
                sum
            }
            WeightFunction::Hermite => {
                let gh = GaussHermite::new(self.order).unwrap();
                let rule = gh.rule();
                let mut sum = 0.0;
                for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
                    sum += weight * f(*node);
                }
                sum
            }
            WeightFunction::ChebyshevI => {
                let gc = GaussChebyshevFirstKind::new(self.order).unwrap();
                let rule = gc.rule();
                let mut sum = 0.0;
                for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
                    sum += weight * f(*node);
                }
                sum
            }
            WeightFunction::ChebyshevII => {
                let gc = GaussChebyshevSecondKind::new(self.order).unwrap();
                let rule = gc.rule();
                let mut sum = 0.0;
                for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
                    sum += weight * f(*node);
                }
                sum
            }
            WeightFunction::LogWeight => {
                // ∫₀¹ -ln(x) · f(x) dx via substitution x = e^{-t}:
                //   = ∫₀^∞ t · e^{-t} · f(e^{-t}) dt
                // This is a Gauss-Laguerre integral with α = 1.
                let gl = GaussLaguerre::new(self.order, 1.0).unwrap();
                let rule = gl.rule();
                let mut sum = 0.0;
                for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
                    sum += weight * f((-*node).exp());
                }
                sum
            }
        }
    }

    /// Integrate f(x) · w(x) over \[a, b\] via affine transform.
    ///
    /// Only applicable for finite-domain weights (Jacobi, ChebyshevI/II).
    /// For LogWeight, maps (0, 1\] to (a, b\] with the log singularity at a.
    /// For Laguerre/Hermite, a and b are ignored (uses natural domain).
    pub fn integrate_over<G>(&self, a: f64, b: f64, f: G) -> f64
    where
        G: Fn(f64) -> f64,
    {
        match &self.weight {
            WeightFunction::Jacobi { .. }
            | WeightFunction::ChebyshevI
            | WeightFunction::ChebyshevII => {
                // Affine map: x in [a,b] <-> t in [-1,1]
                // x = (b-a)/2 * t + (a+b)/2
                // w(x) dx in terms of t includes a Jacobian factor
                // For Jacobi weight: (1-t)^a (1+t)^b maps directly
                let half = 0.5 * (b - a);
                let mid = 0.5 * (a + b);
                self.integrate(|t| f(half * t + mid)) * half.powi(1)
            }
            WeightFunction::LogWeight => {
                // Map (0,1] to (a,b]: x = a + (b-a)*u, log singularity at a
                // ∫ₐᵇ -ln(x-a) f(x) dx ... actually the weight is -ln(u) on (0,1]
                // For general [a,b]: ∫ₐᵇ -ln(x-a) f(x) dx = (b-a) ∫₀¹ -ln((b-a)u) f(a+(b-a)u) du
                // = (b-a) ∫₀¹ [-ln(b-a) - ln(u)] f(a+(b-a)u) du
                // This doesn't decompose cleanly. Keep it simple: just scale.
                let width = b - a;
                let inner = self.integrate(|u| f(a + width * u));
                width * inner
            }
            // Laguerre and Hermite use natural domains
            WeightFunction::Laguerre { .. } | WeightFunction::Hermite => self.integrate(f),
        }
    }
}

/// Convenience: integrate f(x) · w(x) over the natural domain.
pub fn weighted_integrate<G>(
    f: G,
    weight: WeightFunction,
    order: usize,
) -> Result<f64, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    let wi = WeightedIntegrator::new(weight, order)?;
    Ok(wi.integrate(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_weight_sum() {
        // ∫₋₁¹ (1-x)^0.5 (1+x)^0.5 dx = π/2
        let result = weighted_integrate(
            |_| 1.0,
            WeightFunction::Jacobi {
                alpha: 0.5,
                beta: 0.5,
            },
            20,
        )
        .unwrap();
        assert!(
            (result - core::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "result={result}"
        );
    }

    #[test]
    fn laguerre_constant() {
        // ∫₀^∞ e^(-x) dx = 1
        let result =
            weighted_integrate(|_| 1.0, WeightFunction::Laguerre { alpha: 0.0 }, 10).unwrap();
        assert!((result - 1.0).abs() < 1e-10, "result={result}");
    }

    #[test]
    fn laguerre_linear() {
        // ∫₀^∞ x · e^(-x) dx = Γ(2) = 1
        let result =
            weighted_integrate(|x| x, WeightFunction::Laguerre { alpha: 0.0 }, 10).unwrap();
        assert!((result - 1.0).abs() < 1e-10, "result={result}");
    }

    #[test]
    fn hermite_constant() {
        // ∫₋∞^∞ e^(-x²) dx = √π
        let result = weighted_integrate(|_| 1.0, WeightFunction::Hermite, 10).unwrap();
        assert!(
            (result - core::f64::consts::PI.sqrt()).abs() < 1e-10,
            "result={result}"
        );
    }

    #[test]
    fn chebyshev_i_constant() {
        // ∫₋₁¹ 1/√(1-x²) dx = π
        let result = weighted_integrate(|_| 1.0, WeightFunction::ChebyshevI, 20).unwrap();
        assert!(
            (result - core::f64::consts::PI).abs() < 1e-12,
            "result={result}"
        );
    }

    #[test]
    fn chebyshev_ii_constant() {
        // ∫₋₁¹ √(1-x²) dx = π/2
        let result = weighted_integrate(|_| 1.0, WeightFunction::ChebyshevII, 20).unwrap();
        assert!(
            (result - core::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "result={result}"
        );
    }

    #[test]
    fn log_weight_constant() {
        // ∫₀¹ -ln(x) dx = 1
        let result = weighted_integrate(|_| 1.0, WeightFunction::LogWeight, 20).unwrap();
        assert!((result - 1.0).abs() < 1e-10, "result={result}");
    }

    #[test]
    fn log_weight_linear() {
        // ∫₀¹ -ln(x) · x dx = 1/4
        let result = weighted_integrate(|x| x, WeightFunction::LogWeight, 20).unwrap();
        assert!((result - 0.25).abs() < 1e-10, "result={result}");
    }

    #[test]
    fn log_weight_quadratic() {
        // ∫₀¹ -ln(x) · x² dx = 2/27 (integration by parts twice)
        // Actually: ∫₀¹ -ln(x) x^n dx = 1/(n+1)^2
        // For n=2: 1/9... wait, let me recalculate.
        // ∫₀¹ -ln(x) x² dx = [-x³/3 ln(x)]₀¹ + ∫₀¹ x²/3 dx = 0 + 1/9
        let result = weighted_integrate(|x| x * x, WeightFunction::LogWeight, 20).unwrap();
        assert!(
            (result - 1.0 / 9.0).abs() < 1e-10,
            "result={result}, expected={}",
            1.0 / 9.0
        );
    }

    #[test]
    fn zero_order() {
        assert!(weighted_integrate(|_| 1.0, WeightFunction::Hermite, 0).is_err());
    }
}
