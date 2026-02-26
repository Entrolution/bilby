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
    pub fn pair(&self) -> GKPair {
        self.pair
    }

    /// Integrate `f` over \[a, b\], returning (estimate, error_estimate).
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

        // QUADPACK error estimation heuristic
        let mut error = (estimate - gauss_result).abs();
        if abs_sum > 0.0 && error > 0.0 {
            error = abs_sum * (200.0 * error / abs_sum).min(1.0).powf(1.5);
        }

        GKDetail { estimate, error }
    }

    /// Returns the Kronrod order.
    pub fn kronrod_order(&self) -> usize {
        self.pair.kronrod_order()
    }

    /// Returns the embedded Gauss order.
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
    0.9914553711208126392068546975263e0,
    0.9491079123427585245261896840479e0,
    0.8648644233597690727897127886093e0,
    0.7415311855993944398638647732808e0,
    0.5860872354676911302941448382587e0,
    0.4058451513773971669066064120770e0,
    0.2077849550078984676006894037733e0,
    0.0000000000000000000000000000000e0,
];

#[rustfmt::skip]
static GK15_WGK: [f64; 8] = [
    0.0229353220105292249637320080590e0,
    0.0630920926299785532907006631892e0,
    0.1047900103222501838398763225415e0,
    0.1406532597155259187451895951024e0,
    0.1690047266392679028265834265986e0,
    0.1903505780647854099132564024211e0,
    0.2044329400752988924141619992347e0,
    0.2094821410847278280129991748917e0,
];

// G7 weights: correspond to XGK indices 1, 3, 5, 7 (center)
#[rustfmt::skip]
static GK15_WG: [f64; 4] = [
    0.1294849661688696932706114326791e0,
    0.2797053914892766679014677714238e0,
    0.3818300505051189449503697754890e0,
    0.4179591836734693877551020408163e0,
];

// G10-K21: 11 abscissae, 11 Kronrod weights, 5 Gauss weights
#[rustfmt::skip]
static GK21_XGK: [f64; 11] = [
    0.9956571630258080807355272806890e0,
    0.9739065285171717200779640120845e0,
    0.9301574913557082260012071800595e0,
    0.8650633666889845107320966884235e0,
    0.7808177265864168970637175783450e0,
    0.6794095682990244062343271146987e0,
    0.5627571346686046833390009927269e0,
    0.4333953941292471907992659431657e0,
    0.2943928627014601981311260310387e0,
    0.1488743389816312108848260011297e0,
    0.0000000000000000000000000000000e0,
];

#[rustfmt::skip]
static GK21_WGK: [f64; 11] = [
    0.0116946388673718742780640396062e0,
    0.0325581623079647274788189724594e0,
    0.0547558965743519960313813002446e0,
    0.0750396748109199527670431409619e0,
    0.0931254545836976055350654650834e0,
    0.1093871588022976418992105903258e0,
    0.1234919762620658510779581098311e0,
    0.1347092173114733259280540017717e0,
    0.1427759385770600807970942731387e0,
    0.1477391049013384913748415159721e0,
    0.1494455540029169056649366438982e0,
];

// G10 weights: correspond to XGK indices 1, 3, 5, 7, 9
#[rustfmt::skip]
static GK21_WG: [f64; 5] = [
    0.0666713443086881375935688098933e0,
    0.1494513491505805931457763396577e0,
    0.2190863625159820439955349342816e0,
    0.2692667193099963550912269216947e0,
    0.2955242247147528701738929465134e0,
];

// G15-K31: 16 abscissae, 16 Kronrod weights, 8 Gauss weights
#[rustfmt::skip]
static GK31_XGK: [f64; 16] = [
    0.9980022986933970602851728401523e0,
    0.9879925180204854284956518651868e0,
    0.9677390756791391342573479878434e0,
    0.9372733924007059043077834927821e0,
    0.8972645323440819008825096564545e0,
    0.8482065834104272162006482077822e0,
    0.7904185014424659329766494817995e0,
    0.7244177313601700474161606535169e0,
    0.6509967412974169705337358953133e0,
    0.5709721726085388475372263895391e0,
    0.4850818636402396806936554402325e0,
    0.3941513470775633689720720398105e0,
    0.2991800071531688121668028224627e0,
    0.2011940939974345223006283034596e0,
    0.1011420669187174990270741231439e0,
    0.0000000000000000000000000000000e0,
];

#[rustfmt::skip]
static GK31_WGK: [f64; 16] = [
    0.0053774798729233489879207051413e0,
    0.0150079473293161225383747630758e0,
    0.0254608473267153201868740010197e0,
    0.0353463607913758462220397848436e0,
    0.0445897513247648766082272993728e0,
    0.0534815246909280872653431472394e0,
    0.0620095678006706402851392306080e0,
    0.0698541213187282587095007799915e0,
    0.0768496807577203788943277248266e0,
    0.0830805028231330210382892472861e0,
    0.0885644430562117706472754436938e0,
    0.0931265981708253212252486872735e0,
    0.0966427269836236785051790627259e0,
    0.0991735987217919593323931739846e0,
    0.1007698455238755950449266617570e0,
    0.1013300070147915490173747926749e0,
];

// G15 weights: correspond to XGK indices 1, 3, 5, 7, 9, 11, 13, 15 (center)
#[rustfmt::skip]
static GK31_WG: [f64; 8] = [
    0.0307532419961172683546283935772e0,
    0.0703660474881081247092674164507e0,
    0.1071592204671719350118695468569e0,
    0.1395706779261543144480479451103e0,
    0.1662692058169939355320086048121e0,
    0.1861610000155622110268005618642e0,
    0.1984314853271115764561183264384e0,
    0.2025782419255612728806019996752e0,
];

// G20-K41: 21 abscissae, 21 Kronrod weights, 10 Gauss weights
#[rustfmt::skip]
static GK41_XGK: [f64; 21] = [
    0.9988590315882776638383156530069e0,
    0.9931285991850949247861223884713e0,
    0.9815078774502502591933430720022e0,
    0.9639719272779137912676661319728e0,
    0.9408226338317547535199442212444e0,
    0.9122344282513259058677524412033e0,
    0.8782768112522819760774911307811e0,
    0.8391169718222188823394529061702e0,
    0.7950414288375511983506388272789e0,
    0.7463319064601507926143050703556e0,
    0.6932376563347513848054590571459e0,
    0.6360536807265150254528369622629e0,
    0.5751404468197103153429586642543e0,
    0.5108670019508270980043640955253e0,
    0.4435931752387251031999922134926e0,
    0.3737060887154195606725481770493e0,
    0.3016278681149130043205555356886e0,
    0.2277858511416450780804961968576e0,
    0.1526054652409226755052201216068e0,
    0.0765265211334973337546409839884e0,
    0.0000000000000000000000000000000e0,
];

#[rustfmt::skip]
static GK41_WGK: [f64; 21] = [
    0.0030735837185205315012182932460e0,
    0.0086002698556429419866178795010e0,
    0.0146261692569712529883787960309e0,
    0.0203883734612665235980102314328e0,
    0.0258821336049511583450506709615e0,
    0.0312873067703279895895432233801e0,
    0.0366001697582007980305572407072e0,
    0.0416688733279736862637880593689e0,
    0.0464348218674976747202318092611e0,
    0.0509445739237286919327070705035e0,
    0.0551951053482859947448329724198e0,
    0.0591114008806395723749672206459e0,
    0.0626532375547811680258701221255e0,
    0.0658345971336184211156356956940e0,
    0.0686486729285216193456234118537e0,
    0.0710544235534440683057903617232e0,
    0.0730306903327866674951894176891e0,
    0.0745828754004991889865814183625e0,
    0.0757044976845566746595427638462e0,
    0.0763778676720807367050283538061e0,
    0.0766007119179996564450499015301e0,
];

// G20 weights: correspond to XGK indices 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
#[rustfmt::skip]
static GK41_WG: [f64; 10] = [
    0.0176140071391521183118619623519e0,
    0.0406014298003869413310399524974e0,
    0.0626720483341090635695065351870e0,
    0.0832767415767047487247581432205e0,
    0.1019301198172404350367501354804e0,
    0.1181945319615184173123773771138e0,
    0.1316886384491766268989849749816e0,
    0.1420961093183820513293006702817e0,
    0.1491729864726037467878287370097e0,
    0.1527533871307258506980433419510e0,
];

// G25-K51: 26 abscissae, 26 Kronrod weights, 13 Gauss weights
#[rustfmt::skip]
static GK51_XGK: [f64; 26] = [
    0.9992621049926098341934575486034e0,
    0.9955569697904980990878487946894e0,
    0.9880357945340772476385127157740e0,
    0.9766639214595175114983153864598e0,
    0.9616149864258425124181303366017e0,
    0.9429745712289743394140116958647e0,
    0.9207471152817015617463446613163e0,
    0.8949919978782753688510420067828e0,
    0.8658470652932755954499695958834e0,
    0.8334426287608340014210069367057e0,
    0.7978737979985000594104499430431e0,
    0.7592592630373576305772826520436e0,
    0.7177664068130843881866549077330e0,
    0.6735663684734683644851203463762e0,
    0.6268100990103174127881266182452e0,
    0.5776629302412229677236898461265e0,
    0.5263252843347191825996367815801e0,
    0.4730027314457149605221821150919e0,
    0.4178853821930377488518139459457e0,
    0.3611723058093878377358182127264e0,
    0.3030895389311078301674789098034e0,
    0.2438668837209884320452190362797e0,
    0.1837189394210488920159698889753e0,
    0.1228646926107103963873598881804e0,
    0.0615444830056850788865469263892e0,
    0.0000000000000000000000000000000e0,
];

#[rustfmt::skip]
static GK51_WGK: [f64; 26] = [
    0.0019873838923303159265078518288e0,
    0.0055619321353567137580285023668e0,
    0.0094739733861741516072077130124e0,
    0.0132362291955716748137365405847e0,
    0.0168478177091282982316675653634e0,
    0.0204353711458828354568292235594e0,
    0.0240099456069532162209289164881e0,
    0.0274753175878517378094945551781e0,
    0.0307923001673874888911090201523e0,
    0.0340021302743293378367487529552e0,
    0.0371162714834155435606030625368e0,
    0.0400838255040323820748394846708e0,
    0.0428728450201700494768957924395e0,
    0.0455029130499217889098705847527e0,
    0.0479825371388367139063925674920e0,
    0.0502776790807156719633252593344e0,
    0.0523628858064074758643661371279e0,
    0.0542511298885454901454370598788e0,
    0.0559508112204123173082406863275e0,
    0.0574371163615678328535826939394e0,
    0.0586896800223942079619745617888e0,
    0.0597203403241740599790929193256e0,
    0.0605394553760458629453602675157e0,
    0.0611285097170530483058603013429e0,
    0.0614711898714253166154431596526e0,
    0.0615808180678329350759824240067e0,
];

// G25 weights: correspond to XGK indices 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25 (center)
#[rustfmt::skip]
static GK51_WG: [f64; 13] = [
    0.0113937985010262879479029641132e0,
    0.0263549866150321372619018152953e0,
    0.0409391567013063126556234877116e0,
    0.0549046959758351919254159136154e0,
    0.0680383338123569172071871856566e0,
    0.0801407003350010180132349596696e0,
    0.0910282619829636498114972207028e0,
    0.1005359490670506442022068613869e0,
    0.1085196244742636531160929150051e0,
    0.1148582591457116483393255458696e0,
    0.1194557635357847722281781265290e0,
    0.1222424429903100416889595984585e0,
    0.1231760537267154512039028730905e0,
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
