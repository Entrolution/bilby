//! Halton low-discrepancy sequence.
//!
//! Quasi-random sequence using radical-inverse functions with prime bases.
//! Simpler than Sobol but less uniform in high dimensions (d > ~20).

use crate::error::QuadratureError;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// First 100 primes for Halton sequence bases.
const PRIMES: [u32; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

/// Halton sequence generator.
///
/// Produces quasi-random points in \[0, 1)^d using radical-inverse
/// functions with prime bases.
///
/// # Example
///
/// ```
/// use bilby::cubature::halton::HaltonSequence;
///
/// let mut hal = HaltonSequence::new(2).unwrap();
/// let mut point = [0.0; 2];
/// hal.next_point(&mut point);
/// assert!((point[0] - 0.5).abs() < 1e-14);   // radical inverse of 1 in base 2
/// assert!((point[1] - 1.0/3.0).abs() < 1e-14); // radical inverse of 1 in base 3
/// ```
pub struct HaltonSequence {
    dim: usize,
    bases: Vec<u32>,
    index: u64,
}

impl HaltonSequence {
    /// Create a new Halton sequence generator for `dim` dimensions.
    ///
    /// Supports up to 100 dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`QuadratureError::InvalidInput`] if `dim` is zero or exceeds 100.
    pub fn new(dim: usize) -> Result<Self, QuadratureError> {
        if dim == 0 {
            return Err(QuadratureError::InvalidInput("dimension must be >= 1"));
        }
        if dim > PRIMES.len() {
            return Err(QuadratureError::InvalidInput(
                "Halton sequence supports at most 100 dimensions",
            ));
        }
        Ok(Self {
            dim,
            bases: PRIMES[..dim].to_vec(),
            index: 0,
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
        for (j, p) in point.iter_mut().enumerate().take(self.dim) {
            *p = radical_inverse(self.index, self.bases[j]);
        }
    }

    /// Current index (number of points generated so far).
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

/// Radical-inverse function: reverse the digits of `n` in the given `base`.
fn radical_inverse(mut n: u64, base: u32) -> f64 {
    let base_f = f64::from(base);
    let base_u64 = u64::from(base);
    let mut result = 0.0;
    let mut factor = 1.0 / base_f;
    while n > 0 {
        // n % base_u64 is always < base (a small u32), so fits exactly in f64.
        #[allow(clippy::cast_precision_loss)]
        let digit = (n % base_u64) as f64;
        result += digit * factor;
        n /= base_u64;
        factor /= base_f;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn first_few_points() {
        let mut hal = HaltonSequence::new(2).unwrap();
        let mut pt = [0.0; 2];

        hal.next_point(&mut pt);
        assert!((pt[0] - 0.5).abs() < 1e-14); // 1/2
        assert!((pt[1] - 1.0 / 3.0).abs() < 1e-14); // 1/3

        hal.next_point(&mut pt);
        assert!((pt[0] - 0.25).abs() < 1e-14); // 1/4
        assert!((pt[1] - 2.0 / 3.0).abs() < 1e-14); // 2/3

        hal.next_point(&mut pt);
        assert!((pt[0] - 0.75).abs() < 1e-14); // 3/4
        assert!((pt[1] - 1.0 / 9.0).abs() < 1e-14); // 1/9
    }

    #[test]
    fn points_in_unit_cube() {
        let mut hal = HaltonSequence::new(5).unwrap();
        let mut pt = vec![0.0; 5];
        for _ in 0..100 {
            hal.next_point(&mut pt);
            for &x in &pt {
                assert!(x >= 0.0 && x < 1.0, "x={x} out of [0,1)");
            }
        }
    }

    #[test]
    fn invalid_dim() {
        assert!(HaltonSequence::new(0).is_err());
        assert!(HaltonSequence::new(101).is_err());
    }
}
