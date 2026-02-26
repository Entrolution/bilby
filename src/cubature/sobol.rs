//! Sobol low-discrepancy sequence.
//!
//! Quasi-random sequence using direction numbers from Joe & Kuo (2010).
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
/// from Joe & Kuo (2010). The first dimension uses the Van der Corput
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
            *d = 1u32 << (BITS - 1 - i as u32);
        }

        // Higher dimensions: from direction number table
        for j in 1..dim {
            let entry = &SOBOL_TABLE[j - 1];
            let s = entry.degree;
            let a = entry.coeffs;

            // Initial direction numbers from the table
            for i in 0..s as usize {
                direction[j * b + i] = entry.m[i] << (BITS - 1 - i as u32);
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
    pub fn next_point(&mut self, point: &mut [f64]) {
        assert!(point.len() >= self.dim);
        self.index += 1;

        // Find the rightmost zero bit of (index - 1) (gray-code position)
        let c = (self.index - 1).trailing_ones() as usize;
        let b = BITS as usize;
        let norm = 1.0 / (1u64 << BITS) as f64;

        for (j, p) in point.iter_mut().enumerate().take(self.dim) {
            self.state[j] ^= self.direction[j * b + c];
            *p = self.state[j] as f64 * norm;
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
    pub fn index(&self) -> u64 {
        self.index
    }

    /// Spatial dimension.
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
    /// Initial direction numbers m_1, ..., m_s.
    m: [u32; 8],
}

/// Direction numbers from Joe & Kuo (2010) for dimensions 2..40.
/// Each entry: (degree s, polynomial coefficients a, initial m-values).
///
/// Source: https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-old.1111
static SOBOL_TABLE: [SobolEntry; 39] = [
    SobolEntry {
        degree: 1,
        coeffs: 0,
        m: [1, 0, 0, 0, 0, 0, 0, 0],
    },
    SobolEntry {
        degree: 1,
        coeffs: 1,
        m: [1, 0, 0, 0, 0, 0, 0, 0],
    },
    SobolEntry {
        degree: 2,
        coeffs: 1,
        m: [1, 1, 0, 0, 0, 0, 0, 0],
    },
    SobolEntry {
        degree: 3,
        coeffs: 1,
        m: [1, 3, 1, 0, 0, 0, 0, 0],
    },
    SobolEntry {
        degree: 3,
        coeffs: 2,
        m: [1, 1, 1, 0, 0, 0, 0, 0],
    },
    SobolEntry {
        degree: 4,
        coeffs: 1,
        m: [1, 1, 3, 3, 0, 0, 0, 0],
    },
    SobolEntry {
        degree: 4,
        coeffs: 4,
        m: [1, 3, 5, 13, 0, 0, 0, 0],
    },
    SobolEntry {
        degree: 5,
        coeffs: 2,
        m: [1, 1, 5, 5, 17, 0, 0, 0],
    },
    SobolEntry {
        degree: 5,
        coeffs: 4,
        m: [1, 1, 5, 5, 5, 0, 0, 0],
    },
    SobolEntry {
        degree: 5,
        coeffs: 7,
        m: [1, 1, 7, 11, 19, 0, 0, 0],
    },
    SobolEntry {
        degree: 5,
        coeffs: 11,
        m: [1, 1, 5, 1, 1, 0, 0, 0],
    },
    SobolEntry {
        degree: 5,
        coeffs: 13,
        m: [1, 1, 1, 3, 11, 0, 0, 0],
    },
    SobolEntry {
        degree: 5,
        coeffs: 14,
        m: [1, 3, 5, 5, 31, 0, 0, 0],
    },
    SobolEntry {
        degree: 6,
        coeffs: 1,
        m: [1, 3, 3, 9, 7, 49, 0, 0],
    },
    SobolEntry {
        degree: 6,
        coeffs: 13,
        m: [1, 1, 1, 15, 21, 21, 0, 0],
    },
    SobolEntry {
        degree: 6,
        coeffs: 16,
        m: [1, 3, 1, 13, 27, 49, 0, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 19,
        m: [1, 1, 1, 15, 7, 5, 127, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 22,
        m: [1, 3, 3, 5, 19, 33, 65, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 25,
        m: [1, 3, 7, 11, 29, 17, 85, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 37,
        m: [1, 1, 3, 7, 23, 55, 41, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 41,
        m: [1, 3, 5, 1, 15, 17, 63, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 50,
        m: [1, 1, 7, 9, 31, 29, 17, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 55,
        m: [1, 3, 7, 7, 21, 61, 119, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 59,
        m: [1, 1, 5, 3, 5, 49, 89, 0],
    },
    SobolEntry {
        degree: 7,
        coeffs: 62,
        m: [1, 3, 1, 1, 11, 3, 117, 0],
    },
    SobolEntry {
        degree: 8,
        coeffs: 14,
        m: [1, 3, 3, 3, 15, 17, 17, 193],
    },
    SobolEntry {
        degree: 8,
        coeffs: 52,
        m: [1, 1, 3, 7, 31, 29, 67, 45],
    },
    SobolEntry {
        degree: 8,
        coeffs: 56,
        m: [1, 3, 1, 7, 23, 59, 55, 165],
    },
    SobolEntry {
        degree: 8,
        coeffs: 67,
        m: [1, 3, 5, 9, 21, 37, 7, 51],
    },
    SobolEntry {
        degree: 8,
        coeffs: 69,
        m: [1, 1, 7, 5, 17, 13, 81, 251],
    },
    SobolEntry {
        degree: 8,
        coeffs: 70,
        m: [1, 1, 3, 15, 29, 47, 49, 143],
    },
    SobolEntry {
        degree: 8,
        coeffs: 79,
        m: [1, 3, 7, 3, 21, 39, 29, 45],
    },
    SobolEntry {
        degree: 8,
        coeffs: 81,
        m: [1, 1, 5, 7, 29, 7, 37, 67],
    },
    SobolEntry {
        degree: 8,
        coeffs: 84,
        m: [1, 1, 5, 5, 11, 57, 97, 175],
    },
    SobolEntry {
        degree: 8,
        coeffs: 87,
        m: [1, 3, 3, 13, 29, 25, 29, 15],
    },
    SobolEntry {
        degree: 8,
        coeffs: 90,
        m: [1, 1, 7, 7, 31, 37, 105, 189],
    },
    SobolEntry {
        degree: 8,
        coeffs: 97,
        m: [1, 3, 5, 3, 25, 37, 5, 187],
    },
    SobolEntry {
        degree: 8,
        coeffs: 103,
        m: [1, 3, 1, 7, 21, 43, 81, 37],
    },
    SobolEntry {
        degree: 8,
        coeffs: 115,
        m: [1, 1, 3, 5, 1, 45, 47, 77],
    },
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
}
