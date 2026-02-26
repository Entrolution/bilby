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
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::ZeroOrder`] if `n` is zero.
    pub fn new(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }

        let (nodes, weights) = compute_gl_pair(n);
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
    }

    /// Create a new n-point Gauss-Legendre rule with parallel node generation.
    ///
    /// Only beneficial for large n (> ~1000) where the Bogaert asymptotic
    /// path is used. For small n, the overhead of thread scheduling exceeds
    /// the computation time.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::ZeroOrder`] if `n` is zero.
    #[cfg(feature = "parallel")]
    pub fn new_par(n: usize) -> Result<Self, QuadratureError> {
        if n == 0 {
            return Err(QuadratureError::ZeroOrder);
        }

        let (nodes, weights) = compute_gl_pair_par(n);
        Ok(Self {
            rule: QuadratureRule { nodes, weights },
        })
    }
}

impl_rule_accessors!(GaussLegendre, nodes_doc: "Returns the nodes on \\[-1, 1\\].");

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
///   `theta_k` ≈ pi * (4k - 1) / (4n + 2)
#[allow(clippy::many_single_char_names)] // n, m, k, x, w are conventional in quadrature
#[allow(clippy::cast_precision_loss)] // n is a quadrature order, always small enough for exact f64
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

/// Evaluate the Legendre polynomial `P_n(x)` and its derivative `P_n'(x)`
/// using the three-term recurrence.
#[allow(clippy::cast_precision_loss)] // n and k are quadrature orders, always small enough for exact f64
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
     2.404_825_557_695_772_768_621_631_879_33,   5.520_078_110_286_310_649_596_604_112_81,
     8.653_727_912_911_012_216_954_198_712_66,  11.791_534_439_014_281_613_743_044_911_9,
    14.930_917_708_487_785_947_762_593_997_4,  18.071_063_967_910_922_543_147_882_975_6,
    21.211_636_629_879_258_959_078_393_350_5,  24.352_471_530_749_302_737_057_944_763_2,
    27.493_479_132_040_254_795_877_288_234_6,  30.634_606_468_431_975_117_549_578_926_9,
    33.775_820_213_573_568_684_238_546_346_7,  36.917_098_353_664_043_979_769_493_063_3,
    40.058_425_764_628_239_294_799_307_374_0,  43.199_791_713_176_730_357_524_072_728_7,
    46.341_188_371_661_814_018_685_788_879_1,  49.482_609_897_397_817_173_602_761_533_2,
    52.624_051_841_114_996_029_251_285_380_4,  55.765_510_755_019_979_311_683_492_773_5,
    58.906_983_926_080_942_132_834_406_634_6,  62.048_469_190_227_169_882_852_500_264_6,
];

// Tabulated J_1(j_{0,k})^2 for k = 1..21
#[rustfmt::skip]
const BESSEL_J1_SQUARED: [f64; 21] = [
    0.269_514_123_941_916_926_139_021_992_911, 0.115_780_138_582_203_695_807_812_836_182,
    0.073_686_351_136_408_215_140_647_681_198_5, 0.054_037_573_198_116_282_041_774_918_275_8,
    0.042_661_429_017_243_091_265_510_606_349_5, 0.035_242_103_490_996_101_358_747_303_364_8,
    0.030_021_070_103_054_672_675_088_815_768_8, 0.026_147_391_495_308_088_590_458_467_539_9,
    0.023_159_121_824_691_392_265_267_638_217_8, 0.020_783_829_122_267_857_603_980_805_729_7,
    0.018_850_450_669_317_667_816_105_680_021_4, 0.017_246_157_569_665_008_299_524_005_354_2,
    0.015_893_518_105_923_597_802_706_559_428_7, 0.014_737_626_096_472_189_589_574_298_259_2,
    0.013_738_465_145_387_117_918_288_048_413_4, 0.012_866_181_737_615_132_879_140_663_722_8,
    0.012_098_051_548_626_797_547_107_543_849_7, 0.011_416_471_224_491_608_516_862_722_298_6,
    0.010_807_592_791_180_204_011_554_728_683_0, 0.010_260_372_926_280_762_811_042_399_279_0,
    0.009_765_897_139_791_050_540_598_467_366_96,
];

/// Compute the k-th zero of `J_0(x)`, k >= 1.
#[allow(clippy::cast_precision_loss)] // k is a node index, always small enough for exact f64
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
                + r2 * (-0.807_291_666_666_666_666_666_666_666_667e-1
                    + r2 * (0.246_028_645_833_333_333_333_333_333_333
                        + r2 * (-1.824_438_767_206_101_190_476_190_476_19
                            + r2 * (25.336_414_797_343_905_009_920_634_920_6
                                + r2 * (-567.644_412_135_183_381_139_802_038_240
                                    + r2 * (18_690.476_528_232_065_383_163_634_506_4
                                        + r2 * (-8.493_535_802_991_487_699_218_769_836_60e5
                                            + 5.092_254_624_022_267_694_986_812_867_58e7
                                                * r2))))))))
    }
}

/// Compute `J_1(j_{0,k})^2`, k >= 1.
#[allow(clippy::cast_precision_loss)] // k is a node index, always small enough for exact f64
fn bessel_j1_squared(k: usize) -> f64 {
    if k <= 21 {
        BESSEL_J1_SQUARED[k - 1]
    } else {
        // Asymptotic expansion
        let x = 1.0 / (k as f64 - 0.25);
        let x2 = x * x;
        x * (0.202_642_367_284_675_542_887_758_926_420
            + x2 * x2
                * (-0.303_380_429_711_290_253_026_202_643_516e-3
                    + x2 * (0.198_924_364_245_969_295_201_137_972_743e-3
                        + x2 * (-0.228_969_902_772_111_653_038_747_229_723e-3
                            + x2 * (0.433_710_719_130_746_277_915_572_905_025e-3
                                + x2 * (-0.123_632_349_727_175_414_724_737_657_367e-2
                                    + x2 * (0.496_101_423_268_883_102_872_271_417_616e-2
                                        + x2 * (-0.266_837_393_702_323_757_700_998_557_826e-1
                                            + 0.185_395_398_206_345_628_711_318_848_386 * x2))))))))
    }
}

/// Compute a single (theta, weight) pair using Bogaert's asymptotic expansion.
/// k is 1-indexed, referring to the k-th node from the left (k <= ceil(n/2)).
#[allow(clippy::cast_precision_loss)] // n is a quadrature order, always small enough for exact f64
fn bogaert_pair(n: usize, k: usize) -> (f64, f64) {
    let w = 1.0 / (n as f64 + 0.5);
    let nu = bessel_j0_zero(k);
    let theta = w * nu;
    let x = theta * theta;

    // Chebyshev interpolants for node correction
    let sf1t = (((((-1.290_529_962_742_805_084_734_679_683_79e-12 * x
        + 2.407_246_858_643_301_218_259_761_751_84e-10)
        * x
        - 3.131_486_546_359_920_414_688_557_400_12e-8)
        * x
        + 0.275_573_168_962_061_235_623_801_563_453e-5)
        * x
        - 0.148_809_523_713_909_147_898_955_880_165e-3)
        * x
        + 0.416_666_666_665_193_394_525_296_923_981e-2)
        * x
        - 0.416_666_666_666_662_959_639_712_457_549e-1;

    let sf2t = (((((2.206_394_217_818_710_037_347_868_843_22e-9 * x
        - 7.530_367_713_737_693_268_110_307_535_38e-8)
        * x
        + 0.161_969_259_453_836_261_731_700_382_098e-5)
        * x
        - 0.253_300_326_008_232_025_914_059_965_302e-4)
        * x
        + 0.282_116_886_057_560_434_805_998_583_817e-3)
        * x
        - 0.209_022_248_387_852_902_722_635_654_229e-2)
        * x
        + 0.815_972_221_772_932_265_640_401_128_517e-2;

    let sf3t = (((((-2.970_582_253_755_262_298_997_819_566_73e-8 * x
        + 5.558_453_302_237_962_096_558_863_257_12e-7)
        * x
        - 0.567_797_841_356_833_081_642_185_432_056e-5)
        * x
        + 0.418_498_100_329_504_574_443_885_193_835e-4)
        * x
        - 0.251_395_293_283_965_914_823_026_348_764e-3)
        * x
        + 0.128_654_198_542_845_137_196_151_147_483e-2)
        * x
        - 0.416_012_165_620_204_364_833_694_266_818e-2;

    // Chebyshev interpolants for weight correction
    let wsf1t = ((((((((-2.209_028_610_446_166_383_985_734_274_75e-14 * x
        + 2.303_657_268_603_773_768_732_325_788_71e-12)
        * x
        - 1.752_577_007_354_238_076_598_510_423_18e-10)
        * x
        + 1.037_560_669_279_167_958_210_980_093_53e-8)
        * x
        - 4.639_686_475_532_213_312_515_296_310_98e-7)
        * x
        + 0.149_644_593_625_028_648_361_395_938_176e-4)
        * x
        - 0.326_278_659_594_412_170_300_449_074_873e-3)
        * x
        + 0.436_507_936_507_598_105_249_726_413_120e-2)
        * x
        - 0.305_555_555_555_553_028_279_487_898_503e-1)
        * x
        + 0.833_333_333_333_333_302_184_063_103_900e-1;

    let wsf2t = (((((((3.631_174_121_526_547_834_559_294_830_29e-12 * x
        + 7.676_435_450_698_931_307_795_018_443_23e-11)
        * x
        - 7.129_128_572_336_422_206_506_431_506_25e-9)
        * x
        + 2.114_838_806_859_471_514_663_701_302_77e-7)
        * x
        - 0.381_817_918_680_045_468_483_009_307_090e-5)
        * x
        + 0.465_969_530_694_968_391_417_927_388_162e-4)
        * x
        - 0.407_297_185_611_335_764_191_683_161_117e-3)
        * x
        + 0.268_959_435_694_729_660_779_984_493_795e-2)
        * x
        - 0.111_111_111_111_214_923_138_249_347_172e-1;

    let wsf3t = (((((((2.018_267_912_567_033_018_066_432_649_22e-9 * x
        - 4.386_471_225_202_066_492_510_632_125_45e-8)
        * x
        + 5.088_983_472_886_716_531_374_510_932_08e-7)
        * x
        - 0.397_933_316_519_135_275_712_977_531_366e-5)
        * x
        + 0.200_559_326_396_458_326_778_521_795_392e-4)
        * x
        - 0.422_888_059_282_921_161_626_339_411_388e-4)
        * x
        - 0.105_646_050_254_076_140_548_678_457_002e-3)
        * x
        - 0.947_969_308_958_577_323_145_923_317_955e-4)
        * x
        + 0.656_966_489_926_484_797_412_985_260_842e-2;

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

/// Compute all n nodes and weights in parallel.
///
/// Uses the same Newton/Bogaert split as the sequential version, but
/// parallelises the per-node computation using rayon.
#[cfg(feature = "parallel")]
#[allow(clippy::many_single_char_names)] // n, m, k, x, w are conventional in quadrature
#[allow(clippy::cast_precision_loss)] // n is a quadrature order, always small enough for exact f64
fn compute_gl_pair_par(n: usize) -> (Vec<f64>, Vec<f64>) {
    use rayon::prelude::*;

    let m = n.div_ceil(2);

    if n <= ASYMPTOTIC_THRESHOLD {
        // Newton path: each node is independent after computing the initial guess
        let pairs: Vec<(usize, f64, f64)> = (0..m)
            .into_par_iter()
            .map(|i| {
                let k = (i + 1) as f64;
                let nf = n as f64;
                let theta = core::f64::consts::PI * (4.0 * k - 1.0) / (4.0 * nf + 2.0);
                let mut x = theta.cos();

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
                (i, x, w)
            })
            .collect();

        let mut nodes = vec![0.0_f64; n];
        let mut weights = vec![0.0_f64; n];
        for (i, x, w) in pairs {
            nodes[n - 1 - i] = x;
            weights[n - 1 - i] = w;
            nodes[i] = -x;
            weights[i] = w;
        }
        (nodes, weights)
    } else {
        // Bogaert path: embarrassingly parallel
        let pairs: Vec<(usize, f64, f64)> = (0..m)
            .into_par_iter()
            .map(|i| {
                let (theta, weight) = bogaert_pair(n, i + 1);
                (i, theta.cos(), weight)
            })
            .collect();

        let mut nodes = vec![0.0_f64; n];
        let mut weights = vec![0.0_f64; n];
        for (i, x, w) in pairs {
            nodes[n - 1 - i] = x;
            weights[n - 1 - i] = w;
            if i != n - 1 - i {
                nodes[i] = -x;
                weights[i] = w;
            }
        }
        (nodes, weights)
    }
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

    /// Parallel node generation matches sequential for both Newton and Bogaert paths.
    #[cfg(feature = "parallel")]
    #[test]
    fn new_par_matches_sequential() {
        for n in [5, 50, 100, 200, 1000] {
            let seq = GaussLegendre::new(n).unwrap();
            let par = GaussLegendre::new_par(n).unwrap();
            for i in 0..n {
                let node_err = (seq.nodes()[i] - par.nodes()[i]).abs();
                let weight_err = (seq.weights()[i] - par.weights()[i]).abs();
                assert!(node_err < 1e-15, "n={n}, i={i}: node diff = {node_err}");
                assert!(
                    weight_err < 1e-15,
                    "n={n}, i={i}: weight diff = {weight_err}"
                );
            }
        }
    }

    /// Parallel composite integration matches sequential.
    #[cfg(feature = "parallel")]
    #[test]
    fn composite_par_matches_sequential() {
        let gl = GaussLegendre::new(5).unwrap();
        let f = |x: f64| x.sin();
        let seq = gl
            .rule()
            .integrate_composite(0.0, core::f64::consts::PI, 100, f);
        let par = gl
            .rule()
            .integrate_composite_par(0.0, core::f64::consts::PI, 100, f);
        assert!((seq - par).abs() < 1e-14, "seq={seq}, par={par}");
    }
}
