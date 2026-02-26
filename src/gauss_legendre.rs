//! Gauss-Legendre quadrature rule.
//!
//! Computes nodes and weights for Gauss-Legendre quadrature using the
//! Bogaert (2014) algorithm for large n (O(1) per node via asymptotic
//! expansions) and Newton iteration on the Legendre recurrence for small n.
//!
//! # References
//!
//! - Bogaert, I. (2014). "Iteration-Free Computation of Gauss-Legendre
//!   Quadrature Nodes and Weights". SIAM J. Sci. Comput. 36(3), A1008-A1026.
#![allow(clippy::excessive_precision)]

use crate::error::QuadratureError;
use crate::rule::QuadratureRule;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// A Gauss-Legendre quadrature rule.
///
/// Exact for polynomials of degree 2n - 1, where n is the number of points.
///
/// # Example
///
/// ```
/// use bilby::GaussLegendre;
///
/// let gl = GaussLegendre::new(10).unwrap();
/// // Integrate x^3 over [-1, 1] (exact result = 0)
/// let result = gl.rule().integrate(-1.0, 1.0, |x: f64| x * x * x);
/// assert!(result.abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct GaussLegendre {
    rule: QuadratureRule<f64>,
}

impl GaussLegendre {
    /// Create a new n-point Gauss-Legendre rule.
    ///
    /// Returns an error if `n == 0`.
    pub fn new(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }

        let (nodes, weights) = compute_gl_pair(n);
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
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

// ---------------------------------------------------------------------------
// Threshold: for n <= this value, use Newton refinement on the three-term
// recurrence. For n > this value, use Bogaert's asymptotic expansion.
// The asymptotics are accurate to machine epsilon for n > ~100.
// We use a conservative crossover.
// ---------------------------------------------------------------------------
const ASYMPTOTIC_THRESHOLD: usize = 100;

/// Compute all n nodes and weights for a Gauss-Legendre rule on [-1, 1].
fn compute_gl_pair(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];

    // Exploit symmetry: only compute ceil(n/2) nodes.
    let m = n.div_ceil(2);

    if n <= ASYMPTOTIC_THRESHOLD {
        compute_newton(n, m, &mut nodes, &mut weights);
    } else {
        compute_bogaert(n, m, &mut nodes, &mut weights);
    }

    (nodes, weights)
}

// ---------------------------------------------------------------------------
// Newton iteration approach for small n
// ---------------------------------------------------------------------------

/// Compute GL nodes/weights via Newton iteration on the Legendre recurrence.
///
/// The initial guess for each root uses the Tricomi approximation:
///   theta_k ≈ pi * (4k - 1) / (4n + 2)
fn compute_newton(n: usize, m: usize, nodes: &mut [f64], weights: &mut [f64]) {
    let nf = n as f64;

    for i in 0..m {
        // Tricomi initial guess (1-indexed: k = i + 1)
        let k = (i + 1) as f64;
        let theta = core::f64::consts::PI * (4.0 * k - 1.0) / (4.0 * nf + 2.0);
        let mut x = theta.cos();

        // Newton iteration on P_n(x) using the three-term recurrence
        for _ in 0..100 {
            let (p_n, p_n_deriv) = legendre_eval(n, x);
            let dx = -p_n / p_n_deriv;
            x += dx;
            if dx.abs() < 2.0 * f64::EPSILON * x.abs().max(1.0) {
                break;
            }
        }

        let (_, p_n_deriv) = legendre_eval(n, x);
        let w = 2.0 / ((1.0 - x * x) * p_n_deriv * p_n_deriv);

        // Store: nodes are in increasing order (-1 to 1).
        // i=0 is the most negative node.
        nodes[n - 1 - i] = x;
        weights[n - 1 - i] = w;

        // Mirror
        nodes[i] = -x;
        weights[i] = w;
    }
}

/// Evaluate the Legendre polynomial P_n(x) and its derivative P_n'(x)
/// using the three-term recurrence.
pub(crate) fn legendre_eval(n: usize, x: f64) -> (f64, f64) {
    let mut p_prev = 1.0; // P_0(x)
    let mut p_curr = x; // P_1(x)

    if n == 0 {
        return (1.0, 0.0);
    }

    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
    }

    // P_n'(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    let nf = n as f64;
    let deriv = nf * (x * p_curr - p_prev) / (x * x - 1.0);

    (p_curr, deriv)
}

// ---------------------------------------------------------------------------
// Bogaert asymptotic expansion for large n
// ---------------------------------------------------------------------------

// Tabulated zeros of J_0(x) for k = 1..20
#[rustfmt::skip]
const BESSEL_J0_ZEROS: [f64; 20] = [
     2.40482555769577276862163187933,   5.52007811028631064959660411281,
     8.65372791291101221695419871266,  11.7915344390142816137430449119,
    14.9309177084877859477625939974,  18.0710639679109225431478829756,
    21.2116366298792589590783933505,  24.3524715307493027370579447632,
    27.4934791320402547958772882346,  30.6346064684319751175495789269,
    33.7758202135735686842385463467,  36.9170983536640439797694930633,
    40.0584257646282392947993073740,  43.1997917131767303575240727287,
    46.3411883716618140186857888791,  49.4826098973978171736027615332,
    52.6240518411149960292512853804,  55.7655107550199793116834927735,
    58.9069839260809421328344066346,  62.0484691902271698828525002646,
];

// Tabulated J_1(j_{0,k})^2 for k = 1..21
#[rustfmt::skip]
const BESSEL_J1_SQUARED: [f64; 21] = [
    0.269514123941916926139021992911, 0.115780138582203695807812836182,
    0.0736863511364082151406476811985, 0.0540375731981162820417749182758,
    0.0426614290172430912655106063495, 0.0352421034909961013587473033648,
    0.0300210701030546726750888157688, 0.0261473914953080885904584675399,
    0.0231591218246913922652676382178, 0.0207838291222678576039808057297,
    0.0188504506693176678161056800214, 0.0172461575696650082995240053542,
    0.0158935181059235978027065594287, 0.0147376260964721895895742982592,
    0.0137384651453871179182880484134, 0.0128661817376151328791406637228,
    0.0120980515486267975471075438497, 0.0114164712244916085168627222986,
    0.0108075927911802040115547286830, 0.0102603729262807628110423992790,
    0.00976589713979105054059846736696,
];

/// Compute the k-th zero of J_0(x), k >= 1.
fn bessel_j0_zero(k: usize) -> f64 {
    if k <= 20 {
        BESSEL_J0_ZEROS[k - 1]
    } else {
        // McMahon's asymptotic expansion for large zeros of J_0
        let z = core::f64::consts::PI * (k as f64 - 0.25);
        let r = 1.0 / z;
        let r2 = r * r;
        z + r
            * (0.125
                + r2 * (-0.807291666666666666666666666667e-1
                    + r2 * (0.246028645833333333333333333333
                        + r2 * (-1.82443876720610119047619047619
                            + r2 * (25.3364147973439050099206349206
                                + r2 * (-567.644412135183381139802038240
                                    + r2 * (18690.4765282320653831636345064
                                        + r2 * (-8.49353580299148769921876983660e5
                                            + 5.09225462402226769498681286758e7 * r2))))))))
    }
}

/// Compute J_1(j_{0,k})^2, k >= 1.
fn bessel_j1_squared(k: usize) -> f64 {
    if k <= 21 {
        BESSEL_J1_SQUARED[k - 1]
    } else {
        // Asymptotic expansion
        let x = 1.0 / (k as f64 - 0.25);
        let x2 = x * x;
        x * (0.202642367284675542887758926420
            + x2 * x2
                * (-0.303380429711290253026202643516e-3
                    + x2 * (0.198924364245969295201137972743e-3
                        + x2 * (-0.228969902772111653038747229723e-3
                            + x2 * (0.433710719130746277915572905025e-3
                                + x2 * (-0.123632349727175414724737657367e-2
                                    + x2 * (0.496101423268883102872271417616e-2
                                        + x2 * (-0.266837393702323757700998557826e-1
                                            + 0.185395398206345628711318848386 * x2))))))))
    }
}

/// Compute a single (theta, weight) pair using Bogaert's asymptotic expansion.
/// k is 1-indexed, referring to the k-th node from the left (k <= ceil(n/2)).
fn bogaert_pair(n: usize, k: usize) -> (f64, f64) {
    let w = 1.0 / (n as f64 + 0.5);
    let nu = bessel_j0_zero(k);
    let theta = w * nu;
    let x = theta * theta;

    // Chebyshev interpolants for node correction
    let sf1t = (((((-1.29052996274280508473467968379e-12 * x
        + 2.40724685864330121825976175184e-10)
        * x
        - 3.13148654635992041468855740012e-8)
        * x
        + 0.275573168962061235623801563453e-5)
        * x
        - 0.148809523713909147898955880165e-3)
        * x
        + 0.416666666665193394525296923981e-2)
        * x
        - 0.416666666666662959639712457549e-1;

    let sf2t =
        (((((2.20639421781871003734786884322e-9 * x - 7.53036771373769326811030753538e-8) * x
            + 0.161969259453836261731700382098e-5)
            * x
            - 0.253300326008232025914059965302e-4)
            * x
            + 0.282116886057560434805998583817e-3)
            * x
            - 0.209022248387852902722635654229e-2)
            * x
            + 0.815972221772932265640401128517e-2;

    let sf3t =
        (((((-2.97058225375526229899781956673e-8 * x + 5.55845330223796209655886325712e-7) * x
            - 0.567797841356833081642185432056e-5)
            * x
            + 0.418498100329504574443885193835e-4)
            * x
            - 0.251395293283965914823026348764e-3)
            * x
            + 0.128654198542845137196151147483e-2)
            * x
            - 0.416012165620204364833694266818e-2;

    // Chebyshev interpolants for weight correction
    let wsf1t = ((((((((-2.20902861044616638398573427475e-14 * x
        + 2.30365726860377376873232578871e-12)
        * x
        - 1.75257700735423807659851042318e-10)
        * x
        + 1.03756066927916795821098009353e-8)
        * x
        - 4.63968647553221331251529631098e-7)
        * x
        + 0.149644593625028648361395938176e-4)
        * x
        - 0.326278659594412170300449074873e-3)
        * x
        + 0.436507936507598105249726413120e-2)
        * x
        - 0.305555555555553028279487898503e-1)
        * x
        + 0.833333333333333302184063103900e-1;

    let wsf2t = (((((((3.63117412152654783455929483029e-12 * x
        + 7.67643545069893130779501844323e-11)
        * x
        - 7.12912857233642220650643150625e-9)
        * x
        + 2.11483880685947151466370130277e-7)
        * x
        - 0.381817918680045468483009307090e-5)
        * x
        + 0.465969530694968391417927388162e-4)
        * x
        - 0.407297185611335764191683161117e-3)
        * x
        + 0.268959435694729660779984493795e-2)
        * x
        - 0.111111111111214923138249347172e-1;

    let wsf3t = (((((((2.01826791256703301806643264922e-9 * x
        - 4.38647122520206649251063212545e-8)
        * x
        + 5.08898347288671653137451093208e-7)
        * x
        - 0.397933316519135275712977531366e-5)
        * x
        + 0.200559326396458326778521795392e-4)
        * x
        - 0.422888059282921161626339411388e-4)
        * x
        - 0.105646050254076140548678457002e-3)
        * x
        - 0.947969308958577323145923317955e-4)
        * x
        + 0.656966489926484797412985260842e-2;

    // Apply the expansions
    let nu_over_sin = nu / theta.sin();
    let b_nu_over_sin = bessel_j1_squared(k) * nu_over_sin;
    let w_inv_sinc = w * w * nu_over_sin;
    let wis2 = w_inv_sinc * w_inv_sinc;

    let theta_refined = w * (nu + theta * w_inv_sinc * (sf1t + wis2 * (sf2t + wis2 * sf3t)));
    let deno = b_nu_over_sin + b_nu_over_sin * wis2 * (wsf1t + wis2 * (wsf2t + wis2 * wsf3t));
    let weight = (2.0 * w) / deno;

    (theta_refined, weight)
}

/// Compute GL nodes/weights via Bogaert's asymptotic expansion.
fn compute_bogaert(n: usize, m: usize, nodes: &mut [f64], weights: &mut [f64]) {
    for i in 0..m {
        let k = i + 1; // 1-indexed from the boundary

        let (theta, weight) = bogaert_pair(n, k);

        // theta is measured from 0, so the node is cos(theta) which is near +1.
        // We need to store in increasing order: most negative first.
        let x = theta.cos();

        // k=1 is closest to x=+1. Store at the right end.
        nodes[n - 1 - i] = x;
        weights[n - 1 - i] = weight;

        // Mirror to the left
        if i != n - 1 - i {
            nodes[i] = -x;
            weights[i] = weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_order() {
        assert!(GaussLegendre::new(0).is_err());
    }

    #[test]
    fn single_point() {
        let gl = GaussLegendre::new(1).unwrap();
        assert_eq!(gl.order(), 1);
        assert!((gl.nodes()[0]).abs() < 1e-15);
        assert!((gl.weights()[0] - 2.0).abs() < 1e-15);
    }

    #[test]
    fn two_points() {
        let gl = GaussLegendre::new(2).unwrap();
        let expected_node = 1.0_f64 / 3.0_f64.sqrt();
        assert!((gl.nodes()[0] - (-expected_node)).abs() < 1e-14);
        assert!((gl.nodes()[1] - expected_node).abs() < 1e-14);
        assert!((gl.weights()[0] - 1.0).abs() < 1e-14);
        assert!((gl.weights()[1] - 1.0).abs() < 1e-14);
    }

    /// Gauss-Legendre with n points is exact for polynomials of degree 2n-1.
    /// Test: integral of x^(2n-1) over [-1, 1] = 0 for odd powers.
    #[test]
    fn exact_for_odd_polynomial() {
        for n in 1..=20 {
            let gl = GaussLegendre::new(n).unwrap();
            let deg = 2 * n - 1;
            let result = gl.rule().integrate(-1.0, 1.0, |x: f64| x.powi(deg as i32));
            assert!(
                result.abs() < 1e-12,
                "n={n}, deg={deg}: integral of x^{deg} = {result}, expected 0"
            );
        }
    }

    /// Test: integral of x^(2n-2) over [-1, 1] = 2/(2n-1) for even powers.
    #[test]
    fn exact_for_even_polynomial() {
        for n in 1..=20 {
            let gl = GaussLegendre::new(n).unwrap();
            let deg = if n > 1 { 2 * n - 2 } else { 0 };
            let expected = 2.0 / (deg as f64 + 1.0);
            let result = gl.rule().integrate(-1.0, 1.0, |x: f64| x.powi(deg as i32));
            let err = (result - expected).abs();
            assert!(
                err < 1e-12,
                "n={n}, deg={deg}: got {result}, expected {expected}, err={err}"
            );
        }
    }

    /// Test that weights sum to 2 (the integral of 1 over [-1, 1]).
    #[test]
    fn weights_sum_to_two() {
        for n in [1, 2, 5, 10, 50, 100, 200, 1000] {
            let gl = GaussLegendre::new(n).unwrap();
            let sum: f64 = gl.weights().iter().sum();
            assert!(
                (sum - 2.0).abs() < 1e-13,
                "n={n}: weight sum = {sum}, expected 2.0"
            );
        }
    }

    /// Test symmetry: nodes should satisfy x[i] = -x[n-1-i].
    #[test]
    fn node_symmetry() {
        for n in [3, 7, 10, 50, 100, 200, 1000] {
            let gl = GaussLegendre::new(n).unwrap();
            for i in 0..n / 2 {
                let err = (gl.nodes()[i] + gl.nodes()[n - 1 - i]).abs();
                assert!(err < 1e-14, "n={n}, i={i}: symmetry error = {err}");
            }
        }
    }

    /// Test weight symmetry: w[i] = w[n-1-i].
    #[test]
    fn weight_symmetry() {
        for n in [3, 7, 10, 50, 100, 200, 1000] {
            let gl = GaussLegendre::new(n).unwrap();
            for i in 0..n / 2 {
                let err = (gl.weights()[i] - gl.weights()[n - 1 - i]).abs();
                assert!(err < 1e-14, "n={n}, i={i}: weight symmetry error = {err}");
            }
        }
    }

    /// Test nodes are strictly increasing and within (-1, 1).
    #[test]
    fn nodes_are_ordered_and_bounded() {
        for n in [1, 2, 5, 20, 100, 500] {
            let gl = GaussLegendre::new(n).unwrap();
            for i in 0..n {
                assert!(
                    gl.nodes()[i] > -1.0 && gl.nodes()[i] < 1.0,
                    "n={n}, i={i}: node {} out of bounds",
                    gl.nodes()[i]
                );
            }
            for i in 1..n {
                assert!(
                    gl.nodes()[i] > gl.nodes()[i - 1],
                    "n={n}: nodes not strictly increasing at i={i}"
                );
            }
        }
    }

    /// Integration test on [0, 1]: integral of sin(x) = 1 - cos(1).
    #[test]
    fn integrate_sin_on_unit_interval() {
        let gl = GaussLegendre::new(20).unwrap();
        let result = gl.rule().integrate(0.0, 1.0, f64::sin);
        let expected = 1.0 - 1.0_f64.cos();
        assert!(
            (result - expected).abs() < 1e-14,
            "got {result}, expected {expected}"
        );
    }

    /// Integration test on [0, pi]: integral of sin(x) = 2.
    #[test]
    fn integrate_sin_on_zero_to_pi() {
        let gl = GaussLegendre::new(20).unwrap();
        let result = gl.rule().integrate(0.0, core::f64::consts::PI, f64::sin);
        assert!((result - 2.0).abs() < 1e-13, "got {result}, expected 2.0");
    }

    /// Test composite integration.
    #[test]
    fn composite_integration() {
        let gl = GaussLegendre::new(3).unwrap();
        // x^6 over [0, 1] = 1/7. Degree 6 needs > 3 points for exactness,
        // but composite with enough panels should converge.
        let result = gl
            .rule()
            .integrate_composite(0.0, 1.0, 10, |x: f64| x.powi(6));
        let expected = 1.0 / 7.0;
        assert!(
            (result - expected).abs() < 1e-8,
            "got {result}, expected {expected}"
        );
    }

    /// Test the Bogaert asymptotic path (n > 100) against polynomial exactness.
    #[test]
    fn bogaert_path_polynomial_exactness() {
        let n = 200;
        let gl = GaussLegendre::new(n).unwrap();
        // x^4 over [-1, 1] = 2/5
        let result = gl.rule().integrate(-1.0, 1.0, |x: f64| x.powi(4));
        assert!(
            (result - 0.4).abs() < 1e-13,
            "n={n}: got {result}, expected 0.4"
        );
    }

    /// Crossover test: verify Newton and Bogaert agree near the boundary.
    #[test]
    fn newton_bogaert_crossover() {
        // n=100 uses Newton, n=101 uses Bogaert. Both should integrate
        // polynomials exactly.
        for n in [99, 100, 101, 102] {
            let gl = GaussLegendre::new(n).unwrap();
            // x^2 over [-1, 1] = 2/3
            let result = gl.rule().integrate(-1.0, 1.0, |x: f64| x * x);
            assert!(
                (result - 2.0 / 3.0).abs() < 1e-13,
                "n={n}: got {result}, expected 2/3"
            );
        }
    }
}
