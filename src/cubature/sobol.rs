//! Sobol low-discrepancy sequence.
//!
//! Quasi-random sequence using direction numbers from Joe & Kuo (2008).
//! Better uniformity than Halton sequences, especially in high dimensions.
//!
//! Uses gray-code enumeration for O(1) per-point generation.

use crate::error::QuadratureError;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Number of bits used for Sobol sequence generation.
const BITS: u32 = 32;

/// Sobol sequence generator.
///
/// Generates quasi-random points in \[0, 1)^d using direction numbers
/// from Joe & Kuo (2008). The first dimension uses the Van der Corput
/// sequence (base 2), higher dimensions use primitive polynomials.
///
/// # Example
///
/// ```
/// use bilby::cubature::sobol::SobolSequence;
///
/// let mut sob = SobolSequence::new(3).unwrap();
/// let mut point = [0.0; 3];
/// sob.next_point(&mut point);
/// assert!(point[0] >= 0.0 && point[0] < 1.0);
/// ```
pub struct SobolSequence {
    dim: usize,
    /// Direction numbers: dim x BITS matrix stored flat.
    /// direction[j * BITS + i] is direction number i for dimension j.
    direction: Vec<u32>,
    /// Gray-code counter.
    index: u64,
    /// Current state per dimension (XOR accumulator).
    state: Vec<u32>,
}

impl SobolSequence {
    /// Create a new Sobol sequence generator for `dim` dimensions.
    ///
    /// Supports up to 40 dimensions (expandable with more direction numbers).
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if `dim` is zero or exceeds 40.
    pub fn new(dim: usize) -> Result<Self, QuadratureError> {
        if dim == 0 {
            return Err(QuadratureError::InvalidInput("dimension must be >= 1"));
        }
        if dim > MAX_DIM {
            return Err(QuadratureError::InvalidInput(
                "Sobol sequence supports at most 40 dimensions",
            ));
        }

        let b = BITS as usize;
        let mut direction = vec![0u32; dim * b];

        // Dimension 0: Van der Corput (base 2)
        for (i, d) in direction.iter_mut().enumerate().take(b) {
            // i < BITS (32), so i as u32 cannot truncate.
            #[allow(clippy::cast_possible_truncation)]
            let i_u32 = i as u32;
            *d = 1u32 << (BITS - 1 - i_u32);
        }

        // Higher dimensions: from direction number table
        for j in 1..dim {
            let entry = &SOBOL_TABLE[j - 1];
            let s = entry.degree;
            let a = entry.coeffs;

            // Initial direction numbers from the table
            for i in 0..s as usize {
                // i < s <= 8, so i as u32 cannot truncate.
                #[allow(clippy::cast_possible_truncation)]
                let i_u32 = i as u32;
                direction[j * b + i] = entry.m[i] << (BITS - 1 - i_u32);
            }

            // Generate remaining direction numbers via recurrence
            for i in s as usize..b {
                let mut v = direction[j * b + i - s as usize] >> s;
                v ^= direction[j * b + i - s as usize];
                for k in 1..s as usize {
                    if (a >> (s as usize - 1 - k)) & 1 == 1 {
                        v ^= direction[j * b + i - k];
                    }
                }
                direction[j * b + i] = v;
            }
        }

        Ok(Self {
            dim,
            direction,
            index: 0,
            state: vec![0u32; dim],
        })
    }

    /// Generate the next point in \[0, 1)^d.
    ///
    /// # Panics
    ///
    /// Panics if `point.len()` is less than the sequence dimension.
    pub fn next_point(&mut self, point: &mut [f64]) {
        assert!(point.len() >= self.dim);
        self.index += 1;

        // Find the rightmost zero bit of (index - 1) (gray-code position)
        let c = (self.index - 1).trailing_ones() as usize;
        let b = BITS as usize;
        // 1u64 << 32 = 4294967296, fits exactly in f64.
        #[allow(clippy::cast_precision_loss)]
        let norm = 1.0 / (1u64 << BITS) as f64;

        for (j, p) in point.iter_mut().enumerate().take(self.dim) {
            self.state[j] ^= self.direction[j * b + c];
            *p = f64::from(self.state[j]) * norm;
        }
    }

    /// Skip to a specific index (for parallel generation).
    ///
    /// After calling `skip(n)`, the next call to `next_point` produces
    /// the (n+1)-th point.
    pub fn skip(&mut self, n: u64) {
        let b = BITS as usize;
        // Reset state
        self.state.fill(0);
        self.index = 0;

        // Compute state for index n using gray code
        let gray = n ^ (n >> 1);
        for bit in 0..BITS {
            if (gray >> bit) & 1 == 1 {
                for j in 0..self.dim {
                    self.state[j] ^= self.direction[j * b + bit as usize];
                }
            }
        }
        self.index = n;
    }

    /// Current index.
    #[must_use]
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Spatial dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Maximum supported dimensions.
const MAX_DIM: usize = 40;

/// Direction number table entry for a single dimension.
struct SobolEntry {
    /// Degree of the primitive polynomial.
    degree: u32,
    /// Coefficients of the primitive polynomial (excluding leading and trailing 1).
    coeffs: u32,
    /// Initial direction numbers `m_1`, ..., `m_s`.
    m: [u32; 8],
}

/// Direction numbers from Joe & Kuo (2008) for dimensions 2..=40.
///
/// `SOBOL_TABLE[k]` holds the parameters for spatial dimension `k + 2` (the
/// first dimension is the parameter-free Van der Corput sequence). Each entry
/// is `(degree s, polynomial coefficient encoding a, initial m-values m_1..m_s)`
/// for a distinct primitive polynomial over GF(2), ordered by increasing degree;
/// every `m_i` is odd with `m_i < 2^i`. There is exactly one primitive polynomial
/// of degree 1, so the degree-1 block has a single entry — two degree-1 rows would
/// produce identical direction numbers and collapse two coordinates onto each other.
///
/// Source: `new-joe-kuo-6.21201` from the authors' repository
/// <https://github.com/joe-kuo/sobol_data> (Joe & Kuo, "Constructing Sobol
/// sequences with better two-dimensional projections", SIAM J. Sci. Comput.
/// 30 (2008) 2635-2654).
static SOBOL_TABLE: [SobolEntry; 39] = [
    SobolEntry {
        degree: 1,
        coeffs: 0,
        m: [1, 0, 0, 0, 0, 0, 0, 0],
    }, // d=2
    SobolEntry {
        degree: 2,
        coeffs: 1,
        m: [1, 3, 0, 0, 0, 0, 0, 0],
    }, // d=3
    SobolEntry {
        degree: 3,
        coeffs: 1,
        m: [1, 3, 1, 0, 0, 0, 0, 0],
    }, // d=4
    SobolEntry {
        degree: 3,
        coeffs: 2,
        m: [1, 1, 1, 0, 0, 0, 0, 0],
    }, // d=5
    SobolEntry {
        degree: 4,
        coeffs: 1,
        m: [1, 1, 3, 3, 0, 0, 0, 0],
    }, // d=6
    SobolEntry {
        degree: 4,
        coeffs: 4,
        m: [1, 3, 5, 13, 0, 0, 0, 0],
    }, // d=7
    SobolEntry {
        degree: 5,
        coeffs: 2,
        m: [1, 1, 5, 5, 17, 0, 0, 0],
    }, // d=8
    SobolEntry {
        degree: 5,
        coeffs: 4,
        m: [1, 1, 5, 5, 5, 0, 0, 0],
    }, // d=9
    SobolEntry {
        degree: 5,
        coeffs: 7,
        m: [1, 1, 7, 11, 19, 0, 0, 0],
    }, // d=10
    SobolEntry {
        degree: 5,
        coeffs: 11,
        m: [1, 1, 5, 1, 1, 0, 0, 0],
    }, // d=11
    SobolEntry {
        degree: 5,
        coeffs: 13,
        m: [1, 1, 1, 3, 11, 0, 0, 0],
    }, // d=12
    SobolEntry {
        degree: 5,
        coeffs: 14,
        m: [1, 3, 5, 5, 31, 0, 0, 0],
    }, // d=13
    SobolEntry {
        degree: 6,
        coeffs: 1,
        m: [1, 3, 3, 9, 7, 49, 0, 0],
    }, // d=14
    SobolEntry {
        degree: 6,
        coeffs: 13,
        m: [1, 1, 1, 15, 21, 21, 0, 0],
    }, // d=15
    SobolEntry {
        degree: 6,
        coeffs: 16,
        m: [1, 3, 1, 13, 27, 49, 0, 0],
    }, // d=16
    SobolEntry {
        degree: 6,
        coeffs: 19,
        m: [1, 1, 1, 15, 7, 5, 0, 0],
    }, // d=17
    SobolEntry {
        degree: 6,
        coeffs: 22,
        m: [1, 3, 1, 15, 13, 25, 0, 0],
    }, // d=18
    SobolEntry {
        degree: 6,
        coeffs: 25,
        m: [1, 1, 5, 5, 19, 61, 0, 0],
    }, // d=19
    SobolEntry {
        degree: 7,
        coeffs: 1,
        m: [1, 3, 7, 11, 23, 15, 103, 0],
    }, // d=20
    SobolEntry {
        degree: 7,
        coeffs: 4,
        m: [1, 3, 7, 13, 13, 15, 69, 0],
    }, // d=21
    SobolEntry {
        degree: 7,
        coeffs: 7,
        m: [1, 1, 3, 13, 7, 35, 63, 0],
    }, // d=22
    SobolEntry {
        degree: 7,
        coeffs: 8,
        m: [1, 3, 5, 9, 1, 25, 53, 0],
    }, // d=23
    SobolEntry {
        degree: 7,
        coeffs: 14,
        m: [1, 3, 1, 13, 9, 35, 107, 0],
    }, // d=24
    SobolEntry {
        degree: 7,
        coeffs: 19,
        m: [1, 3, 1, 5, 27, 61, 31, 0],
    }, // d=25
    SobolEntry {
        degree: 7,
        coeffs: 21,
        m: [1, 1, 5, 11, 19, 41, 61, 0],
    }, // d=26
    SobolEntry {
        degree: 7,
        coeffs: 28,
        m: [1, 3, 5, 3, 3, 13, 69, 0],
    }, // d=27
    SobolEntry {
        degree: 7,
        coeffs: 31,
        m: [1, 1, 7, 13, 1, 19, 1, 0],
    }, // d=28
    SobolEntry {
        degree: 7,
        coeffs: 32,
        m: [1, 3, 7, 5, 13, 19, 59, 0],
    }, // d=29
    SobolEntry {
        degree: 7,
        coeffs: 37,
        m: [1, 1, 3, 9, 25, 29, 41, 0],
    }, // d=30
    SobolEntry {
        degree: 7,
        coeffs: 41,
        m: [1, 3, 5, 13, 23, 1, 55, 0],
    }, // d=31
    SobolEntry {
        degree: 7,
        coeffs: 42,
        m: [1, 3, 7, 3, 13, 59, 17, 0],
    }, // d=32
    SobolEntry {
        degree: 7,
        coeffs: 50,
        m: [1, 3, 1, 3, 5, 53, 69, 0],
    }, // d=33
    SobolEntry {
        degree: 7,
        coeffs: 55,
        m: [1, 1, 5, 5, 23, 33, 13, 0],
    }, // d=34
    SobolEntry {
        degree: 7,
        coeffs: 56,
        m: [1, 1, 7, 7, 1, 61, 123, 0],
    }, // d=35
    SobolEntry {
        degree: 7,
        coeffs: 59,
        m: [1, 1, 7, 9, 13, 61, 49, 0],
    }, // d=36
    SobolEntry {
        degree: 7,
        coeffs: 62,
        m: [1, 3, 3, 5, 3, 55, 33, 0],
    }, // d=37
    SobolEntry {
        degree: 8,
        coeffs: 14,
        m: [1, 3, 1, 15, 31, 13, 49, 245],
    }, // d=38
    SobolEntry {
        degree: 8,
        coeffs: 21,
        m: [1, 3, 5, 15, 31, 59, 63, 97],
    }, // d=39
    SobolEntry {
        degree: 8,
        coeffs: 22,
        m: [1, 3, 1, 11, 11, 11, 77, 249],
    }, // d=40
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_point_dim1() {
        let mut sob = SobolSequence::new(1).unwrap();
        let mut pt = [0.0];
        sob.next_point(&mut pt);
        assert!((pt[0] - 0.5).abs() < 1e-10); // Van der Corput: 1 -> 0.5
    }

    #[test]
    fn points_in_unit_cube() {
        let mut sob = SobolSequence::new(5).unwrap();
        let mut pt = vec![0.0; 5];
        for _ in 0..100 {
            sob.next_point(&mut pt);
            for &x in &pt {
                assert!(x >= 0.0 && x < 1.0, "x={x} out of [0,1)");
            }
        }
    }

    #[test]
    fn skip_and_generate() {
        let mut sob1 = SobolSequence::new(3).unwrap();
        let mut pt1 = vec![0.0; 3];
        // Generate 10 points
        for _ in 0..10 {
            sob1.next_point(&mut pt1);
        }

        // Skip to 9, then generate the 10th
        let mut sob2 = SobolSequence::new(3).unwrap();
        sob2.skip(9);
        let mut pt2 = vec![0.0; 3];
        sob2.next_point(&mut pt2);

        for j in 0..3 {
            assert!(
                (pt1[j] - pt2[j]).abs() < 1e-14,
                "j={j}: {} vs {}",
                pt1[j],
                pt2[j]
            );
        }
    }

    #[test]
    fn invalid_dim() {
        assert!(SobolSequence::new(0).is_err());
        assert!(SobolSequence::new(41).is_err());
    }

    #[test]
    fn high_dim_points_in_unit_cube() {
        let dim = 20;
        let mut sob = SobolSequence::new(dim).unwrap();
        let mut pt = vec![0.0; dim];
        for i in 0..200 {
            sob.next_point(&mut pt);
            for (j, &x) in pt.iter().enumerate() {
                assert!(
                    x >= 0.0 && x < 1.0,
                    "point {i}, dim {j}: x={x} out of [0,1)"
                );
            }
        }
    }

    #[test]
    fn high_dim_30_points_in_unit_cube() {
        let dim = 30;
        let mut sob = SobolSequence::new(dim).unwrap();
        let mut pt = vec![0.0; dim];
        for i in 0..100 {
            sob.next_point(&mut pt);
            for (j, &x) in pt.iter().enumerate() {
                assert!(
                    x >= 0.0 && x < 1.0,
                    "point {i}, dim {j}: x={x} out of [0,1)"
                );
            }
        }
    }

    /// The first emitted points must match the unscrambled Joe-Kuo sequence
    /// (`new-joe-kuo-6.21201`). bilby advances `index` before emitting, so it
    /// drops the index-0 zero point: emitted point `n` equals the reference
    /// sequence's index-`n` point. Reference oracle: SciPy `stats.qmc.Sobol`
    /// with `scramble=false`, which uses the same 6.21201 direction numbers.
    ///
    /// This anchors the dimension-2 direction numbers to an external oracle; it
    /// does NOT guard against the coordinate-collapse regression (which lived in
    /// a later table entry) — that is covered by `all_dimensions_pairwise_distinct`
    /// and `direction_table_structurally_valid`.
    #[test]
    fn dim2_matches_reference_sequence() {
        let expected = [
            [0.5, 0.5],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.375, 0.375],
            [0.875, 0.875],
        ];
        let mut sob = SobolSequence::new(2).unwrap();
        let mut pt = [0.0; 2];
        for (n, exp) in expected.iter().enumerate() {
            sob.next_point(&mut pt);
            for j in 0..2 {
                assert!(
                    (pt[j] - exp[j]).abs() < 1e-12,
                    "point {}, dim {j}: {} vs {}",
                    n + 1,
                    pt[j],
                    exp[j]
                );
            }
        }
    }

    /// No two coordinates may follow the same sequence. Identical direction
    /// numbers for two dimensions collapse their joint projection onto the
    /// diagonal and silently bias any integrand that couples them.
    #[test]
    fn all_dimensions_pairwise_distinct() {
        let dim = MAX_DIM;
        let n = 256;
        let mut sob = SobolSequence::new(dim).unwrap();
        let mut cols: Vec<Vec<f64>> = (0..dim).map(|_| Vec::with_capacity(n)).collect();
        let mut pt = vec![0.0; dim];
        for _ in 0..n {
            sob.next_point(&mut pt);
            for (j, &x) in pt.iter().enumerate() {
                cols[j].push(x);
            }
        }
        for a in 0..dim {
            for b in (a + 1)..dim {
                assert!(cols[a] != cols[b], "dimensions {a} and {b} are identical");
            }
        }
    }

    /// The table must encode a valid set of distinct primitive polynomials over
    /// GF(2): degrees non-decreasing, the per-degree counts matching the number
    /// of primitive polynomials of each degree (1,1,2,2,6,6,18 for degrees 1-7,
    /// plus the first 3 of the 16 degree-8 polynomials for dims 38-40), distinct
    /// (degree, coeffs) pairs, and every initial direction number `m_i` odd with
    /// `m_i < 2^i`. A duplicate or out-of-range entry silently collapses or
    /// degrades the sequence.
    #[test]
    fn direction_table_structurally_valid() {
        let mut counts = [0usize; 9]; // indexed by degree 1..=8
        let mut prev_degree = 0u32;
        for (k, e) in SOBOL_TABLE.iter().enumerate() {
            assert!(e.degree >= prev_degree, "degree decreased at entry {k}");
            prev_degree = e.degree;
            let s = e.degree as usize;
            assert!((1..=8).contains(&s), "degree {s} out of range at entry {k}");
            counts[s] += 1;
            for i in 0..s {
                let m = e.m[i];
                assert!(m % 2 == 1, "m[{i}]={m} not odd at entry {k}");
                assert!(
                    m < (1u32 << (i + 1)),
                    "m[{i}]={m} >= 2^{} at entry {k}",
                    i + 1
                );
            }
            for i in s..8 {
                assert_eq!(e.m[i], 0, "unused m[{i}] nonzero at entry {k}");
            }
        }
        assert_eq!(&counts[1..=8], &[1, 1, 2, 2, 6, 6, 18, 3]);

        // (degree, coeffs) pairs must be pairwise distinct.
        for (k, e) in SOBOL_TABLE.iter().enumerate() {
            for (l, f) in SOBOL_TABLE.iter().enumerate().skip(k + 1) {
                assert!(
                    !(e.degree == f.degree && e.coeffs == f.coeffs),
                    "duplicate polynomial at entries {k} and {l}"
                );
            }
        }
    }

    /// QMC estimate of `∫_{[0,1]^d} x_1 * x_2 dx` (0-indexed dims 1, 2) = 1/4.
    /// If two dimensions collapse to the same sequence, the estimate drifts
    /// toward `E[x^2] = 1/3` instead. (Replaces a former f≡1 test that never
    /// inspected the generated points.)
    #[test]
    fn sobol_couples_dimensions() {
        let dim = 5;
        let n = 4096;
        let mut sob = SobolSequence::new(dim).unwrap();
        let mut pt = vec![0.0; dim];
        let mut sum = 0.0;
        for _ in 0..n {
            sob.next_point(&mut pt);
            sum += pt[1] * pt[2];
        }
        let estimate = sum / n as f64;
        assert!(
            (estimate - 0.25).abs() < 1e-3,
            "estimate={estimate}, expected 0.25"
        );
    }

    #[test]
    fn max_dim_construction() {
        // Verify we can construct the maximum supported dimension (40).
        let sob = SobolSequence::new(40);
        assert!(sob.is_ok());
        let mut sob = sob.unwrap();
        assert_eq!(sob.dim(), 40);

        let mut pt = vec![0.0; 40];
        sob.next_point(&mut pt);
        for (j, &x) in pt.iter().enumerate() {
            assert!(x >= 0.0 && x < 1.0, "dim {j}: x={x} out of [0,1)");
        }
    }
}
