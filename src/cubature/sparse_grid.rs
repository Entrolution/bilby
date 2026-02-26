//! Smolyak sparse grid cubature rules.
//!
//! Constructs a sparse quadrature rule from nested 1D rules using the
//! Smolyak combination technique. For smooth functions, sparse grids
//! achieve polynomial-exact integration with O(n log^{d-1} n) points
//! instead of O(n^d) for tensor products.
//!
//! Uses Clenshaw-Curtis rules as the nested 1D basis (the canonical choice).

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(feature = "std")]
use std::collections::BTreeMap;

use crate::cubature::CubatureRule;
use crate::error::QuadratureError;

/// Choice of nested 1D rule family for sparse grid construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseGridBasis {
    /// Clenshaw-Curtis (recommended). Nodes at Chebyshev extrema.
    ClenshawCurtis,
}

/// A Smolyak sparse grid cubature rule on \[-1, 1\]^d.
///
/// # Example
///
/// ```
/// use bilby::cubature::SparseGrid;
///
/// // 3D sparse grid at level 2
/// let sg = SparseGrid::clenshaw_curtis(3, 2).unwrap();
/// // Weight sum = volume of [-1,1]^3 = 8
/// let sum: f64 = sg.rule().weights().iter().sum();
/// assert!((sum - 8.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct SparseGrid {
    rule: CubatureRule,
    level: usize,
}

impl SparseGrid {
    /// Construct a sparse grid of given dimension and level.
    ///
    /// Level 0 uses a single center point. Higher levels add more points
    /// for greater accuracy. The rule is exact for polynomials of increasing
    /// total degree as the level increases.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if `dim` is zero.
    pub fn new(dim: usize, level: usize, _basis: SparseGridBasis) -> Result<Self, QuadratureError> {
        if dim == 0 {
            return Err(QuadratureError::InvalidInput("dimension must be >= 1"));
        }

        let rule = build_smolyak(dim, level);
        Ok(Self { rule, level })
    }

    /// Construct using Clenshaw-Curtis basis (convenience).
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if `dim` is zero.
    pub fn clenshaw_curtis(dim: usize, level: usize) -> Result<Self, QuadratureError> {
        Self::new(dim, level, SparseGridBasis::ClenshawCurtis)
    }

    /// Returns a reference to the underlying cubature rule.
    #[inline]
    #[must_use]
    pub fn rule(&self) -> &CubatureRule {
        &self.rule
    }

    /// Number of cubature points.
    #[inline]
    #[must_use]
    pub fn num_points(&self) -> usize {
        self.rule.num_points()
    }

    /// Spatial dimension.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.rule.dim()
    }

    /// Smolyak level.
    #[inline]
    #[must_use]
    pub fn level(&self) -> usize {
        self.level
    }
}

/// Map a Smolyak level to the number of CC points.
///
/// Level 0 → 1 point (midpoint only).
/// Level l ≥ 1 → 2^l + 1 points.
fn cc_order(level: usize) -> usize {
    if level == 0 {
        1
    } else {
        (1 << level) + 1
    }
}

/// Compute Clenshaw-Curtis nodes and weights for a given number of points.
///
/// Delegates to the shared implementation in [`crate::clenshaw_curtis`].
fn cc_rule(n: usize) -> (Vec<f64>, Vec<f64>) {
    crate::clenshaw_curtis::compute_clenshaw_curtis(n)
}

/// Quantise a float to an integer key for exact point merging.
///
/// CC nodes are cos(k*pi/n) which are algebraic numbers. Using 48-bit
/// quantisation avoids floating-point comparison issues when merging
/// duplicate points from different tensor products.
// x is in [-1, 1], so x * 2^48 fits in i64. The casts are intentional.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn quantise(x: f64) -> i64 {
    (x * (1i64 << 48) as f64).round() as i64
}

/// Build a Smolyak sparse grid using the combination technique.
///
/// The Smolyak formula:
///   `Q_{q,d}` = Σ_{max(q-d+1,0) ≤ |l|-d ≤ q-d} (-1)^{q-|l|+d} C(d-1, q-|l|+d) · (`Q_{l_1}` ⊗ ... ⊗ `Q_{l_d}`)
///
/// Using 0-indexed levels where `l_j` ≥ 0.
fn build_smolyak(dim: usize, level: usize) -> CubatureRule {
    // Precompute CC rules for each level we'll need
    let max_level = level;
    let cc_rules: Vec<(Vec<f64>, Vec<f64>)> =
        (0..=max_level).map(|l| cc_rule(cc_order(l))).collect();

    // Accumulate points via BTreeMap for exact merging
    // Key: quantised d-dimensional point
    let mut point_map: BTreeMap<Vec<i64>, (Vec<f64>, f64)> = BTreeMap::new();

    // Enumerate all multi-indices l = (l_0, ..., l_{d-1}) with each l_j >= 0
    // and |l| in [max(level, dim) - dim, level]
    // (using the convention where |l| = l_0 + ... + l_{d-1} and level = q-d)
    //
    // Standard Smolyak: sum over |l|_1 = q-d+1 ... q (using 0-indexed levels)
    // where |l|_1 = l_1 + ... + l_d and l_j >= 0
    //
    // Coefficient: (-1)^{q - |l|_1} * C(d-1, q - |l|_1)
    // where q = level + d - 1 (adjusting to 1-indexed convention)

    let q = level; // Smolyak level (0-indexed)

    // In 0-indexed convention: enumerate |l| = sum of l_j for l_j >= 0
    // where |l| ranges from max(q - d + 1, 0) to q.
    // Coefficient: (-1)^{q - |l|} * C(d-1, q - |l|)
    let sum_min = (q + 1).saturating_sub(dim);
    let sum_max = q;

    for s in sum_min..=sum_max {
        let diff = q - s;
        // Binomial coefficients for sparse grid levels are small enough to fit in f64.
        #[allow(clippy::cast_precision_loss)]
        let coeff =
            if diff.is_multiple_of(2) { 1.0 } else { -1.0 } * binomial(dim - 1, diff) as f64;

        if coeff.abs() < 1e-300 {
            continue;
        }

        // Enumerate all multi-indices with d components >= 0 summing to s
        let mut multi_idx = vec![0usize; dim];
        multi_idx[0] = s;

        loop {
            // Process this multi-index
            let orders: Vec<usize> = multi_idx.iter().map(|&l| cc_order(l)).collect();
            let total: usize = orders.iter().product();

            let mut indices = vec![0usize; dim];
            for _ in 0..total {
                let mut w = coeff;
                let mut key = Vec::with_capacity(dim);
                let mut point = Vec::with_capacity(dim);

                for j in 0..dim {
                    let (ref nodes, ref weights) = cc_rules[multi_idx[j]];
                    point.push(nodes[indices[j]]);
                    key.push(quantise(nodes[indices[j]]));
                    w *= weights[indices[j]];
                }

                point_map
                    .entry(key)
                    .and_modify(|(_, existing_w)| *existing_w += w)
                    .or_insert((point, w));

                // Increment tensor product indices
                for j in 0..dim {
                    indices[j] += 1;
                    if indices[j] < orders[j] {
                        break;
                    }
                    indices[j] = 0;
                }
            }

            // Advance to next multi-index with components >= 0 summing to s
            if !next_composition(&mut multi_idx, s) {
                break;
            }
        }
    }

    // Collect into flat arrays, dropping near-zero weights
    let mut pairs: Vec<(Vec<f64>, f64)> = point_map
        .into_values()
        .filter(|(_, w)| w.abs() > 1e-15)
        .collect();

    // Sort for deterministic output
    pairs.sort_by(|a, b| {
        a.0.iter()
            .zip(b.0.iter())
            .find_map(|(x, y)| {
                x.partial_cmp(y)
                    .filter(|o| *o != core::cmp::Ordering::Equal)
            })
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let n = pairs.len();
    let mut nodes_flat = Vec::with_capacity(n * dim);
    let mut weights = Vec::with_capacity(n);

    for (pt, w) in pairs {
        nodes_flat.extend_from_slice(&pt);
        weights.push(w);
    }

    CubatureRule::new(nodes_flat, weights, dim)
}

/// Generate the next weak composition of `s` into `d` non-negative parts.
///
/// Enumerates in reverse-lexicographic order starting from `[s, 0, ..., 0]`.
/// Returns false when all compositions have been enumerated.
fn next_composition(c: &mut [usize], s: usize) -> bool {
    let d = c.len();
    if d <= 1 {
        return false;
    }

    // Find rightmost index j < d-1 with c[j] > 0
    let mut j = d - 2;
    loop {
        if c[j] > 0 {
            break;
        }
        if j == 0 {
            return false;
        }
        j -= 1;
    }

    // Decrement c[j], put all remaining sum into c[j+1], zero the rest
    c[j] -= 1;
    let remainder: usize = s - c[..=j].iter().sum::<usize>();
    c[j + 1] = remainder;
    for val in c.iter_mut().take(d).skip(j + 2) {
        *val = 0;
    }

    true
}

/// Binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_0_single_point() {
        let sg = SparseGrid::clenshaw_curtis(3, 0).unwrap();
        assert_eq!(sg.num_points(), 1);
        // Single center point with weight = 2^d = 8
        let sum: f64 = sg.rule().weights().iter().sum();
        assert!((sum - 8.0).abs() < 1e-14);
    }

    #[test]
    fn invalid_dim() {
        assert!(SparseGrid::clenshaw_curtis(0, 1).is_err());
    }

    /// Weight sum should equal 2^d for [-1,1]^d.
    #[test]
    fn weight_sum() {
        for d in 1..=4 {
            for q in 0..=3 {
                let sg = SparseGrid::clenshaw_curtis(d, q).unwrap();
                let sum: f64 = sg.rule().weights().iter().sum();
                let expected = 2.0_f64.powi(d as i32);
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "d={d}, q={q}: sum={sum}, expected={expected}, n={}",
                    sg.num_points()
                );
            }
        }
    }

    /// 1D sparse grid should match Clenshaw-Curtis.
    #[test]
    fn one_d_matches_cc() {
        let sg = SparseGrid::clenshaw_curtis(1, 3).unwrap();
        // Level 3 CC has 2^3 + 1 = 9 points
        assert_eq!(sg.num_points(), 9);
    }

    /// Point counts for known cases.
    #[test]
    fn point_counts() {
        // Reference: sparse-grids.de for CC-based Smolyak
        // d=2, q=1: 5 points
        let sg = SparseGrid::clenshaw_curtis(2, 1).unwrap();
        assert_eq!(sg.num_points(), 5);

        // d=2, q=2: 13 points
        let sg = SparseGrid::clenshaw_curtis(2, 2).unwrap();
        assert_eq!(sg.num_points(), 13);

        // d=3, q=1: 7 points
        let sg = SparseGrid::clenshaw_curtis(3, 1).unwrap();
        assert_eq!(sg.num_points(), 7);
    }

    /// Polynomial exactness: level q should integrate low-degree polynomials exactly.
    #[test]
    fn polynomial_exactness_2d() {
        let sg = SparseGrid::clenshaw_curtis(2, 3).unwrap();
        // integral of x^2 * y^2 over [-1,1]^2 = (2/3)^2 = 4/9
        let result = sg.rule().integrate(|x| x[0] * x[0] * x[1] * x[1]);
        assert!((result - 4.0 / 9.0).abs() < 1e-12, "result={result}");
    }

    /// Sparse grid is much sparser than tensor product.
    #[test]
    fn sparsity_advantage() {
        // 5D, level 3: sparse grid vs tensor product
        let sg = SparseGrid::clenshaw_curtis(5, 3).unwrap();
        let tp_points = 9usize.pow(5); // CC level 3 = 9 points per dim
        assert!(
            sg.num_points() < tp_points / 10,
            "sparse={} should be much less than tensor={}",
            sg.num_points(),
            tp_points
        );
    }

    /// Integral of a smooth function over [0,1]^3.
    #[test]
    fn smooth_3d_integral() {
        let sg = SparseGrid::clenshaw_curtis(3, 4).unwrap();
        let result = sg
            .rule()
            .integrate_box(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], |x| {
                (x[0] + x[1] + x[2]).exp()
            });
        // Exact: (e-1)^3 ≈ 5.07321...
        let e_minus_1 = core::f64::consts::E - 1.0;
        let expected = e_minus_1 * e_minus_1 * e_minus_1;
        assert!(
            (result - expected).abs() < 1e-6,
            "result={result}, expected={expected}"
        );
    }
}
