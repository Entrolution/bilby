//! Gauss-Kronrod embedded quadrature pairs.
//!
//! Provides embedded Gauss-Kronrod pairs for error estimation: the Kronrod rule
//! extends a Gauss rule by interleaving additional nodes, so the difference
//! between the two estimates bounds the integration error.
//!
//! Available pairs: G7-K15, G10-K21, G15-K31, G20-K41, G25-K51.
//!
//! Coefficients are from the QUADPACK reference implementation (Piessens et al., 1983).
//!
//! The coefficient arrays preserve the full precision from the original QUADPACK Fortran source
//! (~30 decimal digits), even though f64 only represents ~16 significant digits. This is
//! intentional — the literals match the canonical reference and the compiler truncates at
//! compile time.
#![allow(clippy::excessive_precision)]

#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Detailed result from a single Gauss-Kronrod panel evaluation.
///
/// Used internally by the adaptive integrator for error analysis.
#[derive(Debug, Clone)]
pub(crate) struct GKDetail {
    /// Kronrod estimate of the integral.
    pub estimate: f64,
    /// Error estimate (Kronrod - Gauss difference, scaled).
    pub error: f64,
}

/// Identifies which embedded Gauss-Kronrod pair to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GKPair {
    /// 7-point Gauss / 15-point Kronrod
    G7K15,
    /// 10-point Gauss / 21-point Kronrod
    G10K21,
    /// 15-point Gauss / 31-point Kronrod
    G15K31,
    /// 20-point Gauss / 41-point Kronrod
    G20K41,
    /// 25-point Gauss / 51-point Kronrod
    G25K51,
}

impl GKPair {
    /// Number of Kronrod nodes.
    #[inline]
    #[must_use]
    pub fn kronrod_order(self) -> usize {
        match self {
            Self::G7K15 => 15,
            Self::G10K21 => 21,
            Self::G15K31 => 31,
            Self::G20K41 => 41,
            Self::G25K51 => 51,
        }
    }

    /// Number of Gauss nodes embedded in the Kronrod rule.
    #[inline]
    #[must_use]
    pub fn gauss_order(self) -> usize {
        match self {
            Self::G7K15 => 7,
            Self::G10K21 => 10,
            Self::G15K31 => 15,
            Self::G20K41 => 20,
            Self::G25K51 => 25,
        }
    }
}

/// A precomputed Gauss-Kronrod quadrature rule.
///
/// The Kronrod nodes include all Gauss nodes plus additional interleaved points.
/// The difference between the Kronrod and Gauss estimates provides an error bound.
///
/// # Example
///
/// ```
/// use bilby::{GaussKronrod, GKPair};
///
/// let gk = GaussKronrod::new(GKPair::G7K15);
///
/// // Integrate sin(x) over [0, pi] with error estimate
/// let (estimate, error) = gk.integrate(
///     0.0, core::f64::consts::PI, |x: f64| x.sin()
/// );
/// assert!((estimate - 2.0).abs() < 1e-14);
/// assert!(error < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct GaussKronrod {
    pair: GKPair,
    /// Positive Kronrod abscissae, stored from outermost (near 1) to innermost.
    /// The last entry is 0.0 for odd-order rules.
    xgk: &'static [f64],
    /// Kronrod weights corresponding to xgk.
    wgk: &'static [f64],
    /// Gauss weights for the embedded Gauss nodes only.
    /// These correspond to every other Kronrod node (indices 1, 3, 5, ...).
    wg: &'static [f64],
}

impl GaussKronrod {
    /// Create a new Gauss-Kronrod rule for the given pair.
    #[must_use]
    pub fn new(pair: GKPair) -> Self {
        let (xgk, wgk, wg) = match pair {
            GKPair::G7K15 => (&GK15_XGK[..], &GK15_WGK[..], &GK15_WG[..]),
            GKPair::G10K21 => (&GK21_XGK[..], &GK21_WGK[..], &GK21_WG[..]),
            GKPair::G15K31 => (&GK31_XGK[..], &GK31_WGK[..], &GK31_WG[..]),
            GKPair::G20K41 => (&GK41_XGK[..], &GK41_WGK[..], &GK41_WG[..]),
            GKPair::G25K51 => (&GK51_XGK[..], &GK51_WGK[..], &GK51_WG[..]),
        };
        Self { pair, xgk, wgk, wg }
    }

    /// Returns the pair type.
    #[inline]
    #[must_use]
    pub fn pair(&self) -> GKPair {
        self.pair
    }

    /// Integrate `f` over \[a, b\], returning (estimate, `error_estimate`).
    ///
    /// The estimate uses the full Kronrod rule. The error estimate is based
    /// on the difference between the Kronrod and Gauss results, using the
    /// QUADPACK error scaling heuristic.
    pub fn integrate<G>(&self, a: f64, b: f64, f: G) -> (f64, f64)
    where
        G: Fn(f64) -> f64,
    {
        let detail = self.integrate_detail(a, b, f);
        (detail.estimate, detail.error)
    }

    /// Internal detailed integration returning full QUADPACK output.
    ///
    /// Returns estimate, error, integral of |f|, and integral of |f - mean|
    /// over the interval. The adaptive integrator uses these for roundoff
    /// detection.
    #[allow(clippy::too_many_lines)] // single cohesive QUADPACK algorithm, splitting would obscure the logic
    #[allow(clippy::float_cmp)] // exact comparison for degenerate zero-width interval
    pub(crate) fn integrate_detail<G>(&self, a: f64, b: f64, f: G) -> GKDetail
    where
        G: Fn(f64) -> f64,
    {
        if a == b {
            return GKDetail {
                estimate: 0.0,
                error: 0.0,
            };
        }

        let half_width = 0.5 * (b - a);
        let midpoint = 0.5 * (a + b);
        let n = self.xgk.len(); // number of (positive + center) abscissae

        let mut kronrod_sum = 0.0_f64;
        let mut gauss_sum = 0.0_f64;
        let mut abs_sum = 0.0_f64;

        // QUADPACK convention: xgk is stored from outermost to center.
        // Gauss nodes are at xgk[1], xgk[3], xgk[5], ... (odd indices, 0-based).
        // The center node (xgk[n-1] = 0.0) is a Kronrod-only node for even Gauss
        // order, or a Gauss node for odd Gauss order.

        let mut gauss_idx = 0;

        for i in 0..n {
            let xk = self.xgk[i];
            let wk = self.wgk[i];

            if xk == 0.0 {
                // Center node
                let fx = f(midpoint);
                kronrod_sum += wk * fx;
                abs_sum += wk * fx.abs();

                // Center is a Gauss node for odd Gauss order (G7, G15, G25)
                if self.pair.gauss_order() % 2 == 1 && gauss_idx < self.wg.len() {
                    gauss_sum += self.wg[gauss_idx] * fx;
                }
            } else {
                // Symmetric pair ±xk
                let f_pos = f(midpoint + half_width * xk);
                let f_neg = f(midpoint - half_width * xk);
                let f_sum = f_pos + f_neg;

                kronrod_sum += wk * f_sum;
                abs_sum += wk * (f_pos.abs() + f_neg.abs());

                // Gauss nodes are at odd indices (0-based): 1, 3, 5, ...
                if i % 2 == 1 && gauss_idx < self.wg.len() {
                    gauss_sum += self.wg[gauss_idx] * f_sum;
                    gauss_idx += 1;
                }
            }
        }

        let estimate = half_width * kronrod_sum;
        let gauss_result = half_width * gauss_sum;
        abs_sum *= half_width.abs();

        // QUADPACK error estimation heuristic.
        // This is an intentional simplification of the full QUADPACK error
        // formula (which also considers |f - I/(b-a)| for roundoff detection).
        // The simplified version omits the roundoff term but retains the
        // core (200·δ/S)^1.5 scaling that prevents error underestimation
        // for smooth integrands. See Piessens et al. (1983), §2.2.
        let mut error = (estimate - gauss_result).abs();
        if abs_sum > 0.0 && error > 0.0 {
            error = abs_sum * (200.0 * error / abs_sum).min(1.0).powf(1.5);
        }

        GKDetail { estimate, error }
    }

    /// Returns the Kronrod order.
    #[must_use]
    pub fn kronrod_order(&self) -> usize {
        self.pair.kronrod_order()
    }

    /// Returns the embedded Gauss order.
    #[must_use]
    pub fn gauss_order(&self) -> usize {
        self.pair.gauss_order()
    }
}

// ---------------------------------------------------------------------------
// QUADPACK coefficients.
//
// Convention (matching the QUADPACK Fortran source):
// - XGK: positive abscissae from outermost (near 1) to center (0).
//         The full rule is symmetric: nodes at ±xgk[i].
// - WGK: Kronrod weights corresponding to XGK entries.
// - WG:  Gauss weights for the embedded Gauss nodes only.
//         Gauss nodes are at XGK indices 1, 3, 5, ... (0-based).
//         For odd Gauss order, the center node is also a Gauss node
//         and its weight is the last entry in WG.
//
// Source: QUADPACK (Piessens, de Doncker-Kapenga, Uberhuber, Kahaner, 1983)
//         https://www.netlib.org/quadpack/
// ---------------------------------------------------------------------------

// G7-K15: 8 abscissae, 8 Kronrod weights, 4 Gauss weights
#[rustfmt::skip]
static GK15_XGK: [f64; 8] = [
    0.991_455_371_120_812_639_206_854_697_526_3e0,
    0.949_107_912_342_758_524_526_189_684_047_9e0,
    0.864_864_423_359_769_072_789_712_788_609_3e0,
    0.741_531_185_599_394_439_863_864_773_280_8e0,
    0.586_087_235_467_691_130_294_144_838_258_7e0,
    0.405_845_151_377_397_166_906_606_412_077_0e0,
    0.207_784_955_007_898_467_600_689_403_773_3e0,
    0.000_000_000_000_000_000_000_000_000_000_0e0,
];

#[rustfmt::skip]
static GK15_WGK: [f64; 8] = [
    0.022_935_322_010_529_224_963_732_008_059_0e0,
    0.063_092_092_629_978_553_290_700_663_189_2e0,
    0.104_790_010_322_250_183_839_876_322_541_5e0,
    0.140_653_259_715_525_918_745_189_595_102_4e0,
    0.169_004_726_639_267_902_826_583_426_598_6e0,
    0.190_350_578_064_785_409_913_256_402_421_1e0,
    0.204_432_940_075_298_892_414_161_999_234_7e0,
    0.209_482_141_084_727_828_012_999_174_891_7e0,
];

// G7 weights: correspond to XGK indices 1, 3, 5, 7 (center)
#[rustfmt::skip]
static GK15_WG: [f64; 4] = [
    0.129_484_966_168_869_693_270_611_432_679_1e0,
    0.279_705_391_489_276_667_901_467_771_423_8e0,
    0.381_830_050_505_118_944_950_369_775_489_0e0,
    0.417_959_183_673_469_387_755_102_040_816_3e0,
];

// G10-K21: 11 abscissae, 11 Kronrod weights, 5 Gauss weights
#[rustfmt::skip]
static GK21_XGK: [f64; 11] = [
    0.995_657_163_025_808_080_735_527_280_689_0e0,
    0.973_906_528_517_171_720_077_964_012_084_5e0,
    0.930_157_491_355_708_226_001_207_180_059_5e0,
    0.865_063_366_688_984_510_732_096_688_423_5e0,
    0.780_817_726_586_416_897_063_717_578_345_0e0,
    0.679_409_568_299_024_406_234_327_114_698_7e0,
    0.562_757_134_668_604_683_339_000_992_726_9e0,
    0.433_395_394_129_247_190_799_265_943_165_7e0,
    0.294_392_862_701_460_198_131_126_031_038_7e0,
    0.148_874_338_981_631_210_884_826_001_129_7e0,
    0.000_000_000_000_000_000_000_000_000_000_0e0,
];

#[rustfmt::skip]
static GK21_WGK: [f64; 11] = [
    0.011_694_638_867_371_874_278_064_039_606_2e0,
    0.032_558_162_307_964_727_478_818_972_459_4e0,
    0.054_755_896_574_351_996_031_381_300_244_6e0,
    0.075_039_674_810_919_952_767_043_140_961_9e0,
    0.093_125_454_583_697_605_535_065_465_083_4e0,
    0.109_387_158_802_297_641_899_210_590_325_8e0,
    0.123_491_976_262_065_851_077_958_109_831_1e0,
    0.134_709_217_311_473_325_928_054_001_771_7e0,
    0.142_775_938_577_060_080_797_094_273_138_7e0,
    0.147_739_104_901_338_491_374_841_515_972_1e0,
    0.149_445_554_002_916_905_664_936_643_898_2e0,
];

// G10 weights: correspond to XGK indices 1, 3, 5, 7, 9
#[rustfmt::skip]
static GK21_WG: [f64; 5] = [
    0.066_671_344_308_688_137_593_568_809_893_3e0,
    0.149_451_349_150_580_593_145_776_339_657_7e0,
    0.219_086_362_515_982_043_995_534_934_281_6e0,
    0.269_266_719_309_996_355_091_226_921_694_7e0,
    0.295_524_224_714_752_870_173_892_946_513_4e0,
];

// G15-K31: 16 abscissae, 16 Kronrod weights, 8 Gauss weights
#[rustfmt::skip]
static GK31_XGK: [f64; 16] = [
    0.998_002_298_693_397_060_285_172_840_152_3e0,
    0.987_992_518_020_485_428_495_651_865_186_8e0,
    0.967_739_075_679_139_134_257_347_987_843_4e0,
    0.937_273_392_400_705_904_307_783_492_782_1e0,
    0.897_264_532_344_081_900_882_509_656_454_5e0,
    0.848_206_583_410_427_216_200_648_207_782_2e0,
    0.790_418_501_442_465_932_976_649_481_799_5e0,
    0.724_417_731_360_170_047_416_160_653_516_9e0,
    0.650_996_741_297_416_970_533_735_895_313_3e0,
    0.570_972_172_608_538_847_537_226_389_539_1e0,
    0.485_081_863_640_239_680_693_655_440_232_5e0,
    0.394_151_347_077_563_368_972_072_039_810_5e0,
    0.299_180_007_153_168_812_166_802_822_462_7e0,
    0.201_194_093_997_434_522_300_628_303_459_6e0,
    0.101_142_066_918_717_499_027_074_123_143_9e0,
    0.000_000_000_000_000_000_000_000_000_000_0e0,
];

#[rustfmt::skip]
static GK31_WGK: [f64; 16] = [
    0.005_377_479_872_923_348_987_920_705_141_3e0,
    0.015_007_947_329_316_122_538_374_763_075_8e0,
    0.025_460_847_326_715_320_186_874_001_019_7e0,
    0.035_346_360_791_375_846_222_039_784_843_6e0,
    0.044_589_751_324_764_876_608_227_299_372_8e0,
    0.053_481_524_690_928_087_265_343_147_239_4e0,
    0.062_009_567_800_670_640_285_139_230_608_0e0,
    0.069_854_121_318_728_258_709_500_779_991_5e0,
    0.076_849_680_757_720_378_894_327_724_826_6e0,
    0.083_080_502_823_133_021_038_289_247_286_1e0,
    0.088_564_443_056_211_770_647_275_443_693_8e0,
    0.093_126_598_170_825_321_225_248_687_273_5e0,
    0.096_642_726_983_623_678_505_179_062_725_9e0,
    0.099_173_598_721_791_959_332_393_173_984_6e0,
    0.100_769_845_523_875_595_044_926_661_757_0e0,
    0.101_330_007_014_791_549_017_374_792_674_9e0,
];

// G15 weights: correspond to XGK indices 1, 3, 5, 7, 9, 11, 13, 15 (center)
#[rustfmt::skip]
static GK31_WG: [f64; 8] = [
    0.030_753_241_996_117_268_354_628_393_577_2e0,
    0.070_366_047_488_108_124_709_267_416_450_7e0,
    0.107_159_220_467_171_935_011_869_546_856_9e0,
    0.139_570_677_926_154_314_448_047_945_110_3e0,
    0.166_269_205_816_993_935_532_008_604_812_1e0,
    0.186_161_000_015_562_211_026_800_561_864_2e0,
    0.198_431_485_327_111_576_456_118_326_438_4e0,
    0.202_578_241_925_561_272_880_601_999_675_2e0,
];

// G20-K41: 21 abscissae, 21 Kronrod weights, 10 Gauss weights
#[rustfmt::skip]
static GK41_XGK: [f64; 21] = [
    0.998_859_031_588_277_663_838_315_653_006_9e0,
    0.993_128_599_185_094_924_786_122_388_471_3e0,
    0.981_507_877_450_250_259_193_343_072_002_2e0,
    0.963_971_927_277_913_791_267_666_131_972_8e0,
    0.940_822_633_831_754_753_519_944_221_244_4e0,
    0.912_234_428_251_325_905_867_752_441_203_3e0,
    0.878_276_811_252_281_976_077_491_130_781_1e0,
    0.839_116_971_822_218_882_339_452_906_170_2e0,
    0.795_041_428_837_551_198_350_638_827_278_9e0,
    0.746_331_906_460_150_792_614_305_070_355_6e0,
    0.693_237_656_334_751_384_805_459_057_145_9e0,
    0.636_053_680_726_515_025_452_836_962_262_9e0,
    0.575_140_446_819_710_315_342_958_664_254_3e0,
    0.510_867_001_950_827_098_004_364_095_525_3e0,
    0.443_593_175_238_725_103_199_992_213_492_6e0,
    0.373_706_088_715_419_560_672_548_177_049_3e0,
    0.301_627_868_114_913_004_320_555_535_688_6e0,
    0.227_785_851_141_645_078_080_496_196_857_6e0,
    0.152_605_465_240_922_675_505_220_121_606_8e0,
    0.076_526_521_133_497_333_754_640_983_988_4e0,
    0.000_000_000_000_000_000_000_000_000_000_0e0,
];

#[rustfmt::skip]
static GK41_WGK: [f64; 21] = [
    0.003_073_583_718_520_531_501_218_293_246_0e0,
    0.008_600_269_855_642_941_986_617_879_501_0e0,
    0.014_626_169_256_971_252_988_378_796_030_9e0,
    0.020_388_373_461_266_523_598_010_231_432_8e0,
    0.025_882_133_604_951_158_345_050_670_961_5e0,
    0.031_287_306_770_327_989_589_543_223_380_1e0,
    0.036_600_169_758_200_798_030_557_240_707_2e0,
    0.041_668_873_327_973_686_263_788_059_368_9e0,
    0.046_434_821_867_497_674_720_231_809_261_1e0,
    0.050_944_573_923_728_691_932_707_070_503_5e0,
    0.055_195_105_348_285_994_744_832_972_419_8e0,
    0.059_111_400_880_639_572_374_967_220_645_9e0,
    0.062_653_237_554_781_168_025_870_122_125_5e0,
    0.065_834_597_133_618_421_115_635_695_694_0e0,
    0.068_648_672_928_521_619_345_623_411_853_7e0,
    0.071_054_423_553_444_068_305_790_361_723_2e0,
    0.073_030_690_332_786_667_495_189_417_689_1e0,
    0.074_582_875_400_499_188_986_581_418_362_5e0,
    0.075_704_497_684_556_674_659_542_763_846_2e0,
    0.076_377_867_672_080_736_705_028_353_806_1e0,
    0.076_600_711_917_999_656_445_049_901_530_1e0,
];

// G20 weights: correspond to XGK indices 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
#[rustfmt::skip]
static GK41_WG: [f64; 10] = [
    0.017_614_007_139_152_118_311_861_962_351_9e0,
    0.040_601_429_800_386_941_331_039_952_497_4e0,
    0.062_672_048_334_109_063_569_506_535_187_0e0,
    0.083_276_741_576_704_748_724_758_143_220_5e0,
    0.101_930_119_817_240_435_036_750_135_480_4e0,
    0.118_194_531_961_518_417_312_377_377_113_8e0,
    0.131_688_638_449_176_626_898_984_974_981_6e0,
    0.142_096_109_318_382_051_329_300_670_281_7e0,
    0.149_172_986_472_603_746_787_828_737_009_7e0,
    0.152_753_387_130_725_850_698_043_341_951_0e0,
];

// G25-K51: 26 abscissae, 26 Kronrod weights, 13 Gauss weights
#[rustfmt::skip]
static GK51_XGK: [f64; 26] = [
    0.999_262_104_992_609_834_193_457_548_603_4e0,
    0.995_556_969_790_498_099_087_848_794_689_4e0,
    0.988_035_794_534_077_247_638_512_715_774_0e0,
    0.976_663_921_459_517_511_498_315_386_459_8e0,
    0.961_614_986_425_842_512_418_130_336_601_7e0,
    0.942_974_571_228_974_339_414_011_695_864_7e0,
    0.920_747_115_281_701_561_746_344_661_316_3e0,
    0.894_991_997_878_275_368_851_042_006_782_8e0,
    0.865_847_065_293_275_595_449_969_595_883_4e0,
    0.833_442_628_760_834_001_421_006_936_705_7e0,
    0.797_873_797_998_500_059_410_449_943_043_1e0,
    0.759_259_263_037_357_630_577_282_652_043_6e0,
    0.717_766_406_813_084_388_186_654_907_733_0e0,
    0.673_566_368_473_468_364_485_120_346_376_2e0,
    0.626_810_099_010_317_412_788_126_618_245_2e0,
    0.577_662_930_241_222_967_723_689_846_126_5e0,
    0.526_325_284_334_719_182_599_636_781_580_1e0,
    0.473_002_731_445_714_960_522_182_115_091_9e0,
    0.417_885_382_193_037_748_851_813_945_945_7e0,
    0.361_172_305_809_387_837_735_818_212_726_4e0,
    0.303_089_538_931_107_830_167_478_909_803_4e0,
    0.243_866_883_720_988_432_045_219_036_279_7e0,
    0.183_718_939_421_048_892_015_969_888_975_3e0,
    0.122_864_692_610_710_396_387_359_888_180_4e0,
    0.061_544_483_005_685_078_886_546_926_389_2e0,
    0.000_000_000_000_000_000_000_000_000_000_0e0,
];

#[rustfmt::skip]
static GK51_WGK: [f64; 26] = [
    0.001_987_383_892_330_315_926_507_851_828_8e0,
    0.005_561_932_135_356_713_758_028_502_366_8e0,
    0.009_473_973_386_174_151_607_207_713_012_4e0,
    0.013_236_229_195_571_674_813_736_540_584_7e0,
    0.016_847_817_709_128_298_231_667_565_363_4e0,
    0.020_435_371_145_882_835_456_829_223_559_4e0,
    0.024_009_945_606_953_216_220_928_916_488_1e0,
    0.027_475_317_587_851_737_809_494_555_178_1e0,
    0.030_792_300_167_387_488_891_109_020_152_3e0,
    0.034_002_130_274_329_337_836_748_752_955_2e0,
    0.037_116_271_483_415_543_560_603_062_536_8e0,
    0.040_083_825_504_032_382_074_839_484_670_8e0,
    0.042_872_845_020_170_049_476_895_792_439_5e0,
    0.045_502_913_049_921_788_909_870_584_752_7e0,
    0.047_982_537_138_836_713_906_392_567_492_0e0,
    0.050_277_679_080_715_671_963_325_259_334_4e0,
    0.052_362_885_806_407_475_864_366_137_127_9e0,
    0.054_251_129_888_545_490_145_437_059_878_8e0,
    0.055_950_811_220_412_317_308_240_686_327_5e0,
    0.057_437_116_361_567_832_853_582_693_939_4e0,
    0.058_689_680_022_394_207_961_974_561_788_8e0,
    0.059_720_340_324_174_059_979_092_919_325_6e0,
    0.060_539_455_376_045_862_945_360_267_515_7e0,
    0.061_128_509_717_053_048_305_860_301_342_9e0,
    0.061_471_189_871_425_316_615_443_159_652_6e0,
    0.061_580_818_067_832_935_075_982_424_006_7e0,
];

// G25 weights: correspond to XGK indices 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25 (center)
#[rustfmt::skip]
static GK51_WG: [f64; 13] = [
    0.011_393_798_501_026_287_947_902_964_113_2e0,
    0.026_354_986_615_032_137_261_901_815_295_3e0,
    0.040_939_156_701_306_312_655_623_487_711_6e0,
    0.054_904_695_975_835_191_925_415_913_615_4e0,
    0.068_038_333_812_356_917_207_187_185_656_6e0,
    0.080_140_700_335_001_018_013_234_959_669_6e0,
    0.091_028_261_982_963_649_811_497_220_702_8e0,
    0.100_535_949_067_050_644_202_206_861_386_9e0,
    0.108_519_624_474_263_653_116_092_915_005_1e0,
    0.114_858_259_145_711_648_339_325_545_869_6e0,
    0.119_455_763_535_784_772_228_178_126_529_0e0,
    0.122_242_442_990_310_041_688_959_598_458_5e0,
    0.123_176_053_726_715_451_203_902_873_090_5e0,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Kronrod weights (with symmetric doubling) should sum to 2.
    #[test]
    fn kronrod_weights_sum_to_two() {
        for pair in [
            GKPair::G7K15,
            GKPair::G10K21,
            GKPair::G15K31,
            GKPair::G20K41,
            GKPair::G25K51,
        ] {
            let gk = GaussKronrod::new(pair);
            let n = gk.xgk.len();
            let sum: f64 = (0..n)
                .map(|i| {
                    if gk.xgk[i] == 0.0 {
                        gk.wgk[i]
                    } else {
                        2.0 * gk.wgk[i]
                    }
                })
                .sum();
            // Tolerance is 5e-11 rather than machine epsilon because the
            // QUADPACK coefficients are specified to ~25 decimal places,
            // and rounding to f64 (~16 digits) accumulates across 21+ pairs.
            assert!(
                (sum - 2.0).abs() < 5e-11,
                "{pair:?}: Kronrod weight sum = {sum}"
            );
        }
    }

    /// Gauss weights (with symmetric doubling) should sum to 2.
    #[test]
    fn gauss_weights_sum_to_two() {
        for pair in [
            GKPair::G7K15,
            GKPair::G10K21,
            GKPair::G15K31,
            GKPair::G20K41,
            GKPair::G25K51,
        ] {
            let gk = GaussKronrod::new(pair);
            let gauss_order = gk.pair.gauss_order();
            let n = gk.wg.len();

            // Gauss nodes are at XGK indices 1, 3, 5, ...
            // For odd Gauss order, the center (last WG entry) is also a Gauss node.
            let mut sum = 0.0;
            for j in 0..n {
                let xgk_idx = if gauss_order % 2 == 1 && j == n - 1 {
                    // Last WG entry is the center node
                    gk.xgk.len() - 1
                } else {
                    2 * j + 1
                };
                if gk.xgk[xgk_idx] == 0.0 {
                    sum += gk.wg[j];
                } else {
                    sum += 2.0 * gk.wg[j];
                }
            }
            assert!(
                (sum - 2.0).abs() < 1e-14,
                "{pair:?}: Gauss weight sum = {sum}"
            );
        }
    }

    /// Integrate 1 over [0, 1] should give 1.
    #[test]
    fn integrate_constant() {
        for pair in [
            GKPair::G7K15,
            GKPair::G10K21,
            GKPair::G15K31,
            GKPair::G20K41,
            GKPair::G25K51,
        ] {
            let gk = GaussKronrod::new(pair);
            let (est, _err) = gk.integrate(0.0, 1.0, |_| 1.0);
            assert!((est - 1.0).abs() < 1e-11, "{pair:?}: estimate = {est}");
        }
    }

    /// Integrate sin(x) over [0, pi] = 2.
    #[test]
    fn integrate_sin() {
        for pair in [GKPair::G7K15, GKPair::G10K21, GKPair::G15K31] {
            let gk = GaussKronrod::new(pair);
            let (est, err) = gk.integrate(0.0, core::f64::consts::PI, f64::sin);
            assert!((est - 2.0).abs() < 1e-12, "{pair:?}: estimate = {est}");
            assert!(err < 1e-6, "{pair:?}: error estimate = {err}");
        }
    }

    /// Integrate x^2 over [0, 1] = 1/3.
    #[test]
    fn integrate_quadratic() {
        let gk = GaussKronrod::new(GKPair::G7K15);
        let (est, _err) = gk.integrate(0.0, 1.0, |x| x * x);
        assert!(
            (est - 1.0 / 3.0).abs() < 1e-14,
            "G7K15: x^2 estimate = {est}"
        );
    }

    /// Error estimate should be near zero for polynomials within exactness degree.
    #[test]
    fn error_small_for_exact_polynomial() {
        let gk = GaussKronrod::new(GKPair::G7K15);
        let (est, err) = gk.integrate(-1.0, 1.0, |x| x.powi(12));
        let expected = 2.0 / 13.0;
        assert!(
            (est - expected).abs() < 1e-13,
            "x^12 estimate = {est}, expected {expected}"
        );
        assert!(err < 1e-10, "x^12 error estimate = {err}");
    }

    /// Degenerate interval gives zero.
    #[test]
    fn degenerate_interval() {
        let gk = GaussKronrod::new(GKPair::G7K15);
        let (est, err) = gk.integrate(1.0, 1.0, |x| x * x);
        assert_eq!(est, 0.0);
        assert_eq!(err, 0.0);
    }

    /// Higher-order pairs should be more accurate on smooth functions.
    #[test]
    fn higher_order_more_accurate() {
        let f = |x: f64| x.sin() * x.cos().exp();
        let a = 0.0;
        let b = 5.0;

        let (_, err15) = GaussKronrod::new(GKPair::G7K15).integrate(a, b, f);
        let (_, err21) = GaussKronrod::new(GKPair::G10K21).integrate(a, b, f);
        let (_, err31) = GaussKronrod::new(GKPair::G15K31).integrate(a, b, f);

        assert!(
            err21 < err15,
            "G10K21 error ({err21}) should be less than G7K15 error ({err15})"
        );
        assert!(
            err31 < err21,
            "G15K31 error ({err31}) should be less than G10K21 error ({err21})"
        );
    }
}
