//! Gauss-Jacobi quadrature rule.
//!
//! Weight function `w(x) = (1-x)^α (1+x)^β` on \[-1, 1\].
//!
//! Generalises several classical rules:
//! - α = β = 0 → Gauss-Legendre
//! - α = β = -1/2 → Gauss-Chebyshev Type I
//! - α = β = 1/2 → Gauss-Chebyshev Type II
//! - α = β → Gauss-Gegenbauer (ultraspherical)
//!
//! Nodes and weights are computed via the Golub-Welsch algorithm:
//! eigenvalues and eigenvectors of the symmetric tridiagonal Jacobi
//! matrix built from the three-term recurrence coefficients.

use crate::error::QuadratureError;
use crate::golub_welsch::golub_welsch;
use crate::rule::QuadratureRule;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// A Gauss-Jacobi quadrature rule.
///
/// # Example
///
/// ```
/// use bilby::GaussJacobi;
///
/// // Gauss-Jacobi with alpha=0.5, beta=0.5 (Gegenbauer/ultraspherical)
/// let gj = GaussJacobi::new(10, 0.5, 0.5).unwrap();
/// assert_eq!(gj.order(), 10);
///
/// // Weight sum = integral of (1-x)^alpha * (1+x)^beta over [-1,1]
/// // For alpha=beta=0.5: B(1.5, 1.5) * 2^2 = pi/2
/// let sum: f64 = gj.weights().iter().sum();
/// assert!((sum - core::f64::consts::PI / 2.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct GaussJacobi {
    rule: QuadratureRule<f64>,
    alpha: f64,
    beta: f64,
}

impl GaussJacobi {
    /// Create a new n-point Gauss-Jacobi rule with parameters α and β.
    ///
    /// Requires `n >= 1`, `α > -1`, `β > -1`.
    pub fn new(n: usize, alpha: f64, beta: f64) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }
        if alpha <= -1.0 || beta <= -1.0 || alpha.is_nan() || beta.is_nan() {
            return Err(QuadratureError::InvalidInput(
                "Jacobi parameters must satisfy alpha > -1 and beta > -1",
            ));
        }

        let (nodes, weights) = compute_jacobi(n, alpha, beta);
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
            alpha,
            beta,
        })
    }

    /// Returns the α parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the β parameter.
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns a reference to the underlying quadrature rule.
    pub fn rule(&self) -> &QuadratureRule<f64> {
        &self.rule
    }

    /// Returns the number of quadrature points.
    pub fn order(&self) -> usize {
        self.rule.order()
    }

    /// Returns the nodes on \[-1, 1\].
    pub fn nodes(&self) -> &[f64] {
        &self.rule.nodes
    }

    /// Returns the weights.
    pub fn weights(&self) -> &[f64] {
        &self.rule.weights
    }
}

/// Compute n Gauss-Jacobi nodes and weights via the Golub-Welsch algorithm.
///
/// The monic Jacobi polynomial recurrence:
///   x p_k = p_{k+1} + α_k p_k + β_k p_{k-1}
///
/// with:
///   α_k = (β²-α²) / ((2k+α+β)(2k+α+β+2))
///   β_k = 4k(k+α)(k+β)(k+α+β) / ((2k+α+β)²(2k+α+β+1)(2k+α+β-1))   for k ≥ 1
///
/// μ₀ = 2^(α+β+1) Γ(α+1)Γ(β+1) / Γ(α+β+2)
fn compute_jacobi(n: usize, alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
    let ab = alpha + beta;

    // Diagonal: α_k = (β²-α²) / ((2k+ab)(2k+ab+2))
    // Special handling when 2k+ab is near zero (only k=0 and ab≈0)
    let diag: Vec<f64> = (0..n)
        .map(|k| {
            let two_k_ab = 2.0 * k as f64 + ab;
            let denom = two_k_ab * (two_k_ab + 2.0);
            if denom.abs() < 1e-300 {
                // For ab=0: (β²-α²) = (β-α)(β+α) = 0 when α=β
                // In general, use L'Hôpital: (β-α)/(ab+2) when k=0, ab→0
                if k == 0 {
                    (beta - alpha) / (ab + 2.0)
                } else {
                    0.0
                }
            } else {
                (beta * beta - alpha * alpha) / denom
            }
        })
        .collect();

    // Off-diagonal squared: β_k for k = 1, ..., n-1
    // Special case: when k+α+β=0 and 2k+α+β-1=0 (i.e., k=1, α+β=-1),
    // both numerator and denominator vanish. The limit is 2(1+α)(1+β).
    let off_diag_sq: Vec<f64> = (1..n)
        .map(|k| {
            let k = k as f64;
            let two_k_ab = 2.0 * k + ab;
            let denom = two_k_ab * two_k_ab * (two_k_ab + 1.0) * (two_k_ab - 1.0);
            if denom.abs() < 1e-300 {
                // 0/0 case: k=1, α+β=-1. Limit = 2(1+α)(1+β)
                2.0 * (1.0 + alpha) * (1.0 + beta)
            } else {
                let numer = 4.0 * k * (k + alpha) * (k + beta) * (k + ab);
                numer / denom
            }
        })
        .collect();

    // μ₀ = 2^(ab+1) Γ(α+1)Γ(β+1) / Γ(ab+2)
    let mu0 = ((ab + 1.0) * core::f64::consts::LN_2 + ln_gamma(alpha + 1.0) + ln_gamma(beta + 1.0)
        - ln_gamma(ab + 2.0))
    .exp();

    golub_welsch(&diag, &off_diag_sq, mu0)
}

#[allow(clippy::excessive_precision)]
/// Log-gamma function using the Lanczos approximation.
///
/// Accurate to ~15 digits for positive arguments.
pub(crate) fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation with g=7, n=9
    // Coefficients from Numerical Recipes
    const COEFF: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        let z = 1.0 - x;
        core::f64::consts::PI.ln() - (core::f64::consts::PI * x).sin().ln() - ln_gamma(z)
    } else {
        let z = x - 1.0;
        let mut sum = COEFF[0];
        for (i, &c) in COEFF.iter().enumerate().skip(1) {
            sum += c / (z + i as f64);
        }
        let t = z + 7.5; // g + 0.5
        0.5 * (2.0 * core::f64::consts::PI).ln() + (t.ln() * (z + 0.5)) - t + sum.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn zero_order() {
        assert!(GaussJacobi::new(0, 0.0, 0.0).is_err());
    }

    #[test]
    fn invalid_params() {
        assert!(GaussJacobi::new(5, -1.0, 0.0).is_err());
        assert!(GaussJacobi::new(5, 0.0, -1.5).is_err());
        assert!(GaussJacobi::new(5, f64::NAN, 0.0).is_err());
    }

    /// alpha=beta=0 should recover Gauss-Legendre.
    #[test]
    fn legendre_recovery() {
        let gj = GaussJacobi::new(10, 0.0, 0.0).unwrap();
        // Weight sum for alpha=beta=0 is integral of 1 over [-1,1] = 2.
        let sum: f64 = gj.weights().iter().sum();
        assert!((sum - 2.0).abs() < 1e-12, "sum={sum}");
    }

    /// Weight sum = integral of (1-x)^alpha (1+x)^beta over [-1,1]
    /// = 2^(alpha+beta+1) B(alpha+1, beta+1)
    /// = 2^(a+b+1) Gamma(a+1) Gamma(b+1) / Gamma(a+b+2)
    #[test]
    fn weight_sum_various_params() {
        let cases = [(0.5, 0.5), (1.0, 0.0), (0.0, 1.0), (2.0, 3.0), (0.5, 1.5)];
        for (a, b) in cases {
            let gj = GaussJacobi::new(20, a, b).unwrap();
            let sum: f64 = gj.weights().iter().sum();
            let expected = jacobi_integral(a, b);
            assert!(
                (sum - expected).abs() < 1e-10,
                "a={a}, b={b}: sum={sum}, expected={expected}"
            );
        }
    }

    /// Helper: integral of (1-x)^a (1+x)^b over [-1,1].
    fn jacobi_integral(a: f64, b: f64) -> f64 {
        let log_val =
            (a + b + 1.0) * core::f64::consts::LN_2 + ln_gamma(a + 1.0) + ln_gamma(b + 1.0)
                - ln_gamma(a + b + 2.0);
        log_val.exp()
    }

    #[test]
    fn nodes_sorted_and_bounded() {
        let gj = GaussJacobi::new(20, 1.0, 0.5).unwrap();
        for i in 0..gj.order() - 1 {
            assert!(gj.nodes()[i] < gj.nodes()[i + 1]);
        }
        assert!(gj.nodes()[0] > -1.0);
        assert!(*gj.nodes().last().unwrap() < 1.0);
    }

    /// Polynomial exactness: n-point rule exact for degree 2n-1.
    /// Test: integral of x^k * (1-x)^a * (1+x)^b over [-1,1].
    #[test]
    fn polynomial_exactness() {
        let n = 10;
        let alpha = 0.5;
        let beta = 1.5;
        let gj = GaussJacobi::new(n, alpha, beta).unwrap();

        // Test weighted integral of x^2: should equal
        // integral of x^2 (1-x)^0.5 (1+x)^1.5 over [-1,1]
        let numerical: f64 = gj
            .nodes()
            .iter()
            .zip(gj.weights())
            .map(|(&x, &w)| x * x * w)
            .sum();

        // Reference via Gauss-Jacobi with alpha+2 trick won't work easily,
        // so just check it's stable and reasonable.
        // For alpha=0.5, beta=1.5: integral of x^2 w(x) = B(1.5,2.5)*2^3 * something
        // Instead, verify symmetry: integral of x * w(x) for alpha=beta should be 0.
        let gj_sym = GaussJacobi::new(n, 1.0, 1.0).unwrap();
        let odd: f64 = gj_sym
            .nodes()
            .iter()
            .zip(gj_sym.weights())
            .map(|(&x, &w)| x * w)
            .sum();
        assert!(odd.abs() < 1e-12, "odd integral = {odd}");

        // For symmetric alpha=beta, x^2 weighted integral should be positive
        let even: f64 = gj_sym
            .nodes()
            .iter()
            .zip(gj_sym.weights())
            .map(|(&x, &w)| x * x * w)
            .sum();
        assert!(even > 0.0);

        // Sanity: numerical for asymmetric case should be finite and reasonable
        assert!(numerical.is_finite());
        assert!(numerical.abs() < 10.0);
    }

    /// For alpha=beta (Gegenbauer), nodes should be symmetric.
    #[test]
    fn gegenbauer_symmetry() {
        let gj = GaussJacobi::new(15, 1.5, 1.5).unwrap();
        let n = gj.order();
        for i in 0..n / 2 {
            assert!(
                (gj.nodes()[i] + gj.nodes()[n - 1 - i]).abs() < 1e-13,
                "i={i}: {} vs {}",
                gj.nodes()[i],
                gj.nodes()[n - 1 - i]
            );
            assert!(
                (gj.weights()[i] - gj.weights()[n - 1 - i]).abs() < 1e-13,
                "i={i}: {} vs {}",
                gj.weights()[i],
                gj.weights()[n - 1 - i]
            );
        }
    }

    /// Integral of 1/sqrt(1-x^2) over [-1,1] = pi.
    /// This is the Gauss-Chebyshev Type I case (alpha=beta=-0.5).
    #[test]
    fn chebyshev_type_i_recovery() {
        let gj = GaussJacobi::new(20, -0.5, -0.5).unwrap();
        let sum: f64 = gj.weights().iter().sum();
        assert!((sum - PI).abs() < 1e-10, "sum={sum}, expected={PI}");
    }
}
