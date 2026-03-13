//! Oscillatory integration via Filon-Clenshaw-Curtis.
//!
//! Computes ∫ₐᵇ f(x)·sin(ωx) dx or ∫ₐᵇ f(x)·cos(ωx) dx efficiently
//! even when ω is large. Expands f in a Chebyshev basis, then integrates
//! each Chebyshev moment against the oscillatory kernel analytically.
//!
//! For small ω (|ω·(b-a)| < 2), falls back to standard adaptive integration.
//!
//! # Example
//!
//! ```
//! use bilby::oscillatory::integrate_oscillatory_sin;
//!
//! // ∫₀¹ sin(100x) dx = (1 - cos(100)) / 100
//! let exact = (1.0 - 100.0_f64.cos()) / 100.0;
//! let result = integrate_oscillatory_sin(|_| 1.0, 0.0, 1.0, 100.0, 1e-10).unwrap();
//! assert!((result.value - exact).abs() < 1e-8);
//! ```

use crate::adaptive;
use crate::error::QuadratureError;
use crate::gauss_legendre::GaussLegendre;
use crate::result::QuadratureResult;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Type of oscillatory kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OscillatoryKernel {
    /// sin(ω·x)
    Sine,
    /// cos(ω·x)
    Cosine,
}

/// Builder for oscillatory integration via Filon-Clenshaw-Curtis.
///
/// # Example
///
/// ```
/// use bilby::oscillatory::{OscillatoryIntegrator, OscillatoryKernel};
///
/// let integrator = OscillatoryIntegrator::new(OscillatoryKernel::Cosine, 50.0)
///     .with_order(64);
/// let result = integrator.integrate(0.0, 1.0, |_| 1.0).unwrap();
/// let exact = 50.0_f64.sin() / 50.0;
/// assert!((result.value - exact).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct OscillatoryIntegrator {
    kernel: OscillatoryKernel,
    omega: f64,
    order: usize,
    abs_tol: f64,
    rel_tol: f64,
}

impl OscillatoryIntegrator {
    /// Create a new oscillatory integrator.
    #[must_use]
    pub fn new(kernel: OscillatoryKernel, omega: f64) -> Self {
        Self {
            kernel,
            omega,
            order: 32,
            abs_tol: 1.49e-8,
            rel_tol: 1.49e-8,
        }
    }

    /// Set the Chebyshev expansion order.
    #[must_use]
    pub fn with_order(mut self, n: usize) -> Self {
        self.order = n;
        self
    }

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

    /// Integrate f(x)·kernel(ω·x) over \[a, b\].
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::DegenerateInterval`] if `a`, `b`, or `omega`
    /// is non-finite. May also propagate errors from the adaptive fallback when
    /// `|omega * (b - a) / 2|` is small.
    #[allow(clippy::many_single_char_names)] // a, b, f, n are conventional in quadrature
    #[allow(clippy::similar_names)] // sum_c_half / sum_s_half are intentionally parallel names
    #[allow(clippy::cast_precision_loss)] // k, n are small quadrature indices, always exact in f64
    pub fn integrate<G>(
        &self,
        a: f64,
        b: f64,
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(f64) -> f64,
    {
        if !a.is_finite() || !b.is_finite() || !self.omega.is_finite() {
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

        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);
        let theta = self.omega * half;

        // For small theta, fall back to adaptive integration
        if theta.abs() < 2.0 {
            return self.adaptive_fallback(a, b, &f);
        }

        // Filon-Clenshaw-Curtis method
        let n = self.order.max(4);

        // Step 1: Evaluate f at Clenshaw-Curtis nodes on [-1, 1]
        let f_vals: Vec<f64> = (0..=n)
            .map(|k| {
                let t = (k as f64 * core::f64::consts::PI / n as f64).cos();
                f(mid + half * t)
            })
            .collect();
        let num_evals = n + 1;

        // Step 2: Compute Chebyshev coefficients via DCT
        let cheb_coeffs = chebyshev_coefficients(&f_vals, n);

        // Step 3: Compute modified Chebyshev moments
        let (moments_cos, moments_sin) = modified_chebyshev_moments(theta, n);

        // Step 4: Sum I = half * Σ c_j * μ_j
        // For cos kernel: ∫₋₁¹ f(t) cos(θt) dt = Σ c_j μ_j^c
        // For sin kernel: ∫₋₁¹ f(t) sin(θt) dt = Σ c_j μ_j^s
        // Then multiply by half and the phase factor for the shift to [a, b].
        //
        // ∫ₐᵇ f(x) cos(ωx) dx = ∫₋₁¹ f(mid + half·t) cos(ω(mid + half·t)) half dt
        //   = half ∫₋₁¹ f̃(t) [cos(ωmid)cos(θt) - sin(ωmid)sin(θt)] dt
        //   = half [cos(ωmid) Σ cⱼμⱼᶜ - sin(ωmid) Σ cⱼμⱼˢ]

        let sum_c: f64 = cheb_coeffs
            .iter()
            .zip(moments_cos.iter())
            .map(|(c, m)| c * m)
            .sum();
        let sum_s: f64 = cheb_coeffs
            .iter()
            .zip(moments_sin.iter())
            .map(|(c, m)| c * m)
            .sum();

        let omega_mid = self.omega * mid;
        let cos_mid = omega_mid.cos();
        let sin_mid = omega_mid.sin();

        let value = match self.kernel {
            OscillatoryKernel::Cosine => half * (cos_mid * sum_c - sin_mid * sum_s),
            OscillatoryKernel::Sine => half * (sin_mid * sum_c + cos_mid * sum_s),
        };

        // Error estimate: compare with a lower-order approximation (use half the coefficients)
        let n_half = n / 2;
        let sum_c_half: f64 = cheb_coeffs
            .iter()
            .take(n_half + 1)
            .zip(moments_cos.iter())
            .map(|(c, m)| c * m)
            .sum();
        let sum_s_half: f64 = cheb_coeffs
            .iter()
            .take(n_half + 1)
            .zip(moments_sin.iter())
            .map(|(c, m)| c * m)
            .sum();

        let value_half = match self.kernel {
            OscillatoryKernel::Cosine => half * (cos_mid * sum_c_half - sin_mid * sum_s_half),
            OscillatoryKernel::Sine => half * (sin_mid * sum_c_half + cos_mid * sum_s_half),
        };

        let error = (value - value_half).abs();
        let converged = error <= self.abs_tol.max(self.rel_tol * value.abs());

        Ok(QuadratureResult {
            value,
            error_estimate: error,
            num_evals,
            converged,
        })
    }

    /// Fallback to standard adaptive integration for small ω.
    fn adaptive_fallback<G>(
        &self,
        a: f64,
        b: f64,
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(f64) -> f64,
    {
        let omega = self.omega;
        let integrand = match self.kernel {
            OscillatoryKernel::Sine => {
                Box::new(move |x: f64| f(x) * (omega * x).sin()) as Box<dyn Fn(f64) -> f64>
            }
            OscillatoryKernel::Cosine => Box::new(move |x: f64| f(x) * (omega * x).cos()),
        };
        adaptive::adaptive_integrate(&*integrand, a, b, self.abs_tol)
    }
}

/// Compute Chebyshev coefficients from function values at CC nodes.
///
/// Given `f_k` = f(cos(kπ/n)) for k = 0, ..., n, computes the coefficients
/// `c_j` such that f(t) ≈ Σ `c_j` `T_j(t)` (with the convention that `c_0` and
/// `c_n` are halved).
#[allow(clippy::cast_precision_loss)] // j, k, n are small Chebyshev indices, always exact in f64
fn chebyshev_coefficients(f_vals: &[f64], n: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0; n + 1];
    let pi_n = core::f64::consts::PI / n as f64;

    for (j, cj) in coeffs.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (k, fk) in f_vals.iter().enumerate() {
            let factor = if k == 0 || k == n { 0.5 } else { 1.0 };
            sum += factor * fk * (j as f64 * k as f64 * pi_n).cos();
        }
        *cj = 2.0 * sum / n as f64;
    }

    // Halve the first and last coefficients
    coeffs[0] *= 0.5;
    coeffs[n] *= 0.5;

    coeffs
}

/// Compute modified Chebyshev moments via Gauss-Legendre quadrature.
///
/// Returns (`μ_j^c`, `μ_j^s`) for j = 0, ..., n where:
///   `μ_j^c` = ∫₋₁¹ `T_j(x)` cos(θx) dx
///   `μ_j^s` = ∫₋₁¹ `T_j(x)` sin(θx) dx
///
/// Uses GL quadrature with enough nodes to resolve both the Chebyshev
/// polynomial (degree j ≤ n) and the oscillatory kernel (effective
/// frequency ~θ). Converges exponentially for these smooth integrands.
#[allow(clippy::cast_precision_loss)] // j, n are small Chebyshev indices, always exact in f64
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // theta.abs().ceil() is a small positive integer, safe to cast to usize
fn modified_chebyshev_moments(theta: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mc = vec![0.0; n + 1];
    let mut ms = vec![0.0; n + 1];

    if theta.abs() < 1e-15 {
        // cos(θx) ≈ 1, sin(θx) ≈ 0
        // μ_j^c = ∫₋₁¹ T_j(x) dx
        for (j, mcj) in mc.iter_mut().enumerate() {
            if j == 0 {
                *mcj = 2.0;
            } else if j % 2 == 0 {
                *mcj = 2.0 / (1.0 - (j as f64).powi(2));
            }
        }
        return (mc, ms);
    }

    // GL with m nodes is exact for polynomials of degree 2m-1.
    // T_j(x) cos(θx) is smooth and entire; GL converges exponentially
    // when m exceeds the effective bandwidth n + |θ|.
    let m = (n + (theta.abs().ceil() as usize) + 32).max(64);
    let gl = GaussLegendre::new(m).unwrap();
    let rule = gl.rule();

    for (node, weight) in rule.nodes.iter().zip(rule.weights.iter()) {
        let x = *node;
        let cos_tx = (theta * x).cos();
        let sin_tx = (theta * x).sin();

        // Compute T_j(x) via recurrence: T_0=1, T_1=x, T_{k+1}=2xT_k - T_{k-1}
        let mut t_prev = 1.0;
        let mut t_curr = x;

        mc[0] += weight * cos_tx;
        ms[0] += weight * sin_tx;

        if n >= 1 {
            mc[1] += weight * x * cos_tx;
            ms[1] += weight * x * sin_tx;
        }

        for j in 2..=n {
            let t_next = 2.0 * x * t_curr - t_prev;
            mc[j] += weight * t_next * cos_tx;
            ms[j] += weight * t_next * sin_tx;
            t_prev = t_curr;
            t_curr = t_next;
        }
    }

    (mc, ms)
}

/// Convenience: integrate f(x)·sin(ωx) over \[a, b\].
///
/// # Example
///
/// ```
/// use bilby::integrate_oscillatory_sin;
///
/// let exact = (1.0 - 100.0_f64.cos()) / 100.0;
/// let result = integrate_oscillatory_sin(|_| 1.0, 0.0, 1.0, 100.0, 1e-10).unwrap();
/// assert!((result.value - exact).abs() < 1e-8);
/// ```
///
/// # Errors
///
/// Returns [`QuadratureError::DegenerateInterval`] if `a`, `b`, or `omega`
/// is non-finite.
pub fn integrate_oscillatory_sin<G>(
    f: G,
    a: f64,
    b: f64,
    omega: f64,
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    OscillatoryIntegrator::new(OscillatoryKernel::Sine, omega)
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .integrate(a, b, f)
}

/// Convenience: integrate f(x)·cos(ωx) over \[a, b\].
///
/// # Example
///
/// ```
/// use bilby::integrate_oscillatory_cos;
///
/// let exact = 50.0_f64.sin() / 50.0;
/// let result = integrate_oscillatory_cos(|_| 1.0, 0.0, 1.0, 50.0, 1e-10).unwrap();
/// assert!((result.value - exact).abs() < 1e-8);
/// ```
///
/// # Errors
///
/// Returns [`QuadratureError::DegenerateInterval`] if `a`, `b`, or `omega`
/// is non-finite.
pub fn integrate_oscillatory_cos<G>(
    f: G,
    a: f64,
    b: f64,
    omega: f64,
    tol: f64,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(f64) -> f64,
{
    OscillatoryIntegrator::new(OscillatoryKernel::Cosine, omega)
        .with_abs_tol(tol)
        .with_rel_tol(tol)
        .integrate(a, b, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_trivial() {
        // ∫₀^π sin(x) dx = 2  (ω=1, f=1)
        let result =
            integrate_oscillatory_sin(|_| 1.0, 0.0, core::f64::consts::PI, 1.0, 1e-10).unwrap();
        assert!((result.value - 2.0).abs() < 1e-6, "value={}", result.value);
    }

    #[test]
    fn cos_moderate_omega() {
        // ∫₀¹ cos(10x) dx = sin(10)/10
        let exact = 10.0_f64.sin() / 10.0;
        let result = integrate_oscillatory_cos(|_| 1.0, 0.0, 1.0, 10.0, 1e-10).unwrap();
        assert!(
            (result.value - exact).abs() < 1e-8,
            "value={}, exact={exact}",
            result.value
        );
    }

    #[test]
    fn sin_high_omega() {
        // ∫₀¹ sin(100x) dx = (1 - cos(100))/100
        let exact = (1.0 - 100.0_f64.cos()) / 100.0;
        let result = integrate_oscillatory_sin(|_| 1.0, 0.0, 1.0, 100.0, 1e-10).unwrap();
        assert!(
            (result.value - exact).abs() < 1e-8,
            "value={}, exact={exact}",
            result.value
        );
    }

    #[test]
    fn cos_with_linear_f() {
        // ∫₀^π x cos(x) dx = [x sin(x) + cos(x)]₀^π = -1 - 1 = -2
        let result =
            integrate_oscillatory_cos(|x| x, 0.0, core::f64::consts::PI, 1.0, 1e-8).unwrap();
        assert!(
            (result.value - (-2.0)).abs() < 1e-4,
            "value={}",
            result.value
        );
    }

    #[test]
    fn sin_with_exp_f() {
        // ∫₀¹ exp(x) sin(ωx) dx = Im[(e^(1+iω) - 1) / (1+iω)]
        let omega: f64 = 50.0;
        let r = 1.0;
        let i = omega;
        // e^(1+iω) = e * (cos(ω) + i sin(ω))
        let e = core::f64::consts::E;
        let re_num = e * omega.cos() - 1.0;
        let im_num = e * omega.sin();
        let denom = r * r + i * i; // 1 + ω²
                                   // (re_num + i im_num) / (r + i*i_) = (re_num + i im_num)(r - i*i_) / denom
        let exact = (im_num * r - re_num * i) / denom; // imaginary part

        let result = integrate_oscillatory_sin(f64::exp, 0.0, 1.0, omega, 1e-8).unwrap();
        assert!(
            (result.value - exact).abs() < 1e-4,
            "value={}, exact={exact}",
            result.value
        );
    }

    #[test]
    fn small_omega_fallback() {
        // ∫₀¹ cos(0.5x) dx = sin(0.5)/0.5  (falls back to adaptive)
        let exact = 0.5_f64.sin() / 0.5;
        let result = integrate_oscillatory_cos(|_| 1.0, 0.0, 1.0, 0.5, 1e-10).unwrap();
        assert!(
            (result.value - exact).abs() < 1e-8,
            "value={}, exact={exact}",
            result.value
        );
    }

    #[test]
    fn zero_interval() {
        let result = integrate_oscillatory_sin(|_| 1.0, 1.0, 1.0, 10.0, 1e-10).unwrap();
        assert_eq!(result.value, 0.0);
    }

    #[test]
    fn nan_input() {
        assert!(integrate_oscillatory_sin(|_| 1.0, f64::NAN, 1.0, 10.0, 1e-10).is_err());
    }

    #[test]
    fn inf_inputs_rejected() {
        assert!(integrate_oscillatory_sin(|_| 1.0, f64::INFINITY, 1.0, 10.0, 1e-10).is_err());
        assert!(integrate_oscillatory_cos(|_| 1.0, 0.0, f64::NEG_INFINITY, 10.0, 1e-10).is_err());
        assert!(integrate_oscillatory_sin(|_| 1.0, 0.0, 1.0, f64::INFINITY, 1e-10).is_err());
    }

    #[test]
    fn very_high_omega() {
        // ∫₀¹ sin(1000x) dx = (1 - cos(1000))/1000
        let exact = (1.0 - 1000.0_f64.cos()) / 1000.0;
        let result = integrate_oscillatory_sin(|_| 1.0, 0.0, 1.0, 1000.0, 1e-8).unwrap();
        assert!(
            (result.value - exact).abs() < 1e-6,
            "value={}, exact={exact}",
            result.value
        );
    }
}
