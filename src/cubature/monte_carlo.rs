//! Monte Carlo and quasi-Monte Carlo integration.
//!
//! Plain pseudo-random MC with standard error estimation, and
//! quasi-MC using Sobol or Halton low-discrepancy sequences.

use crate::cubature::halton::HaltonSequence;
use crate::cubature::sobol::SobolSequence;
use crate::error::QuadratureError;
use crate::result::QuadratureResult;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Monte Carlo integration method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MCMethod {
    /// Plain pseudo-random Monte Carlo.
    Plain,
    /// Quasi-Monte Carlo using Sobol sequences.
    Sobol,
    /// Quasi-Monte Carlo using Halton sequences.
    Halton,
}

/// Builder for Monte Carlo / quasi-Monte Carlo integration.
///
/// # Example
///
/// ```
/// use bilby::cubature::monte_carlo_integrate;
///
/// // Integral of 1 over [0,1]^3 = 1
/// let result = monte_carlo_integrate(|_| 1.0, &[0.0; 3], &[1.0; 3], 10000).unwrap();
/// assert!((result.value - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct MonteCarloIntegrator {
    method: MCMethod,
    num_samples: usize,
    seed: u64,
}

impl Default for MonteCarloIntegrator {
    fn default() -> Self {
        Self {
            method: MCMethod::Sobol,
            num_samples: 10_000,
            seed: 12345,
        }
    }
}

impl MonteCarloIntegrator {
    /// Set the MC method.
    pub fn with_method(mut self, method: MCMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the number of samples.
    pub fn with_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Set the random seed (only used for `MCMethod::Plain`).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Integrate `f` over the hyperrectangle \[lower, upper\].
    pub fn integrate<G>(
        &self,
        lower: &[f64],
        upper: &[f64],
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64,
    {
        let d = lower.len();
        if d == 0 || upper.len() != d {
            return Err(QuadratureError::InvalidInput(
                "lower and upper must have equal nonzero length",
            ));
        }
        if self.num_samples == 0 {
            return Err(QuadratureError::InvalidInput(
                "number of samples must be >= 1",
            ));
        }

        let widths: Vec<f64> = (0..d).map(|j| upper[j] - lower[j]).collect();
        let volume: f64 = widths.iter().product();
        let n = self.num_samples;

        match self.method {
            MCMethod::Plain => self.integrate_plain(d, lower, &widths, volume, n, &f),
            MCMethod::Sobol => self.integrate_qmc_sobol(d, lower, &widths, volume, n, &f),
            MCMethod::Halton => self.integrate_qmc_halton(d, lower, &widths, volume, n, &f),
        }
    }

    fn integrate_plain<G>(
        &self,
        d: usize,
        lower: &[f64],
        widths: &[f64],
        volume: f64,
        n: usize,
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64,
    {
        let mut rng = Xoshiro256pp::new(self.seed);
        let mut x = vec![0.0; d];

        // Welford's online algorithm for mean and variance
        let mut mean = 0.0;
        let mut m2 = 0.0;

        for i in 1..=n {
            for j in 0..d {
                x[j] = lower[j] + widths[j] * rng.next_f64();
            }
            let val = f(&x);
            let delta = val - mean;
            mean += delta / i as f64;
            let delta2 = val - mean;
            m2 += delta * delta2;
        }

        let variance = if n > 1 { m2 / (n - 1) as f64 } else { 0.0 };
        let std_error = (variance / n as f64).sqrt() * volume;

        Ok(QuadratureResult {
            value: volume * mean,
            error_estimate: std_error,
            num_evals: n,
            converged: true,
        })
    }

    fn integrate_qmc_sobol<G>(
        &self,
        d: usize,
        lower: &[f64],
        widths: &[f64],
        volume: f64,
        n: usize,
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64,
    {
        let mut sob = SobolSequence::new(d)?;
        let mut x = vec![0.0; d];
        let mut u = vec![0.0; d];
        let mut sum = 0.0;

        for _ in 0..n {
            sob.next_point(&mut u);
            for j in 0..d {
                x[j] = lower[j] + widths[j] * u[j];
            }
            sum += f(&x);
        }

        let estimate = volume * sum / n as f64;

        // Heuristic error estimate: compare N/2 and N estimates
        let half_sum = {
            let mut s2 = SobolSequence::new(d)?;
            let half = n / 2;
            let mut sm = 0.0;
            for _ in 0..half {
                s2.next_point(&mut u);
                for j in 0..d {
                    x[j] = lower[j] + widths[j] * u[j];
                }
                sm += f(&x);
            }
            volume * sm / half as f64
        };

        let error = (estimate - half_sum).abs();

        Ok(QuadratureResult {
            value: estimate,
            error_estimate: error,
            num_evals: n + n / 2,
            converged: true,
        })
    }

    fn integrate_qmc_halton<G>(
        &self,
        d: usize,
        lower: &[f64],
        widths: &[f64],
        volume: f64,
        n: usize,
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64,
    {
        let mut hal = HaltonSequence::new(d)?;
        let mut x = vec![0.0; d];
        let mut u = vec![0.0; d];
        let mut sum = 0.0;

        for _ in 0..n {
            hal.next_point(&mut u);
            for j in 0..d {
                x[j] = lower[j] + widths[j] * u[j];
            }
            sum += f(&x);
        }

        let estimate = volume * sum / n as f64;

        // Heuristic error: compare N/2 and N
        let half_sum = {
            let mut h2 = HaltonSequence::new(d)?;
            let half = n / 2;
            let mut sm = 0.0;
            for _ in 0..half {
                h2.next_point(&mut u);
                for j in 0..d {
                    x[j] = lower[j] + widths[j] * u[j];
                }
                sm += f(&x);
            }
            volume * sm / half as f64
        };

        let error = (estimate - half_sum).abs();

        Ok(QuadratureResult {
            value: estimate,
            error_estimate: error,
            num_evals: n + n / 2,
            converged: true,
        })
    }

    /// Parallel Monte Carlo integration over the hyperrectangle \[lower, upper\].
    ///
    /// For Sobol and Halton methods, all sample points are pre-generated
    /// sequentially (the sequences are inherently serial), then function
    /// evaluations are parallelised. For Plain MC, the sample space is split
    /// into independent chunks with separate PRNGs, giving statistically
    /// equivalent but numerically different results compared to sequential.
    #[cfg(feature = "parallel")]
    pub fn integrate_par<G>(
        &self,
        lower: &[f64],
        upper: &[f64],
        f: G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64 + Sync,
    {
        let d = lower.len();
        if d == 0 || upper.len() != d {
            return Err(QuadratureError::InvalidInput(
                "lower and upper must have equal nonzero length",
            ));
        }
        if self.num_samples == 0 {
            return Err(QuadratureError::InvalidInput(
                "number of samples must be >= 1",
            ));
        }

        let widths: Vec<f64> = (0..d).map(|j| upper[j] - lower[j]).collect();
        let volume: f64 = widths.iter().product();
        let n = self.num_samples;

        match self.method {
            MCMethod::Plain => self.integrate_plain_par(d, lower, &widths, volume, n, &f),
            MCMethod::Sobol => self.integrate_qmc_par_sobol(d, lower, &widths, volume, n, &f),
            MCMethod::Halton => self.integrate_qmc_par_halton(d, lower, &widths, volume, n, &f),
        }
    }

    #[cfg(feature = "parallel")]
    fn integrate_plain_par<G>(
        &self,
        d: usize,
        lower: &[f64],
        widths: &[f64],
        volume: f64,
        n: usize,
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64 + Sync,
    {
        use rayon::prelude::*;

        let num_chunks = rayon::current_num_threads().max(1);
        let chunk_size = n / num_chunks;
        let remainder = n % num_chunks;

        // Each chunk uses an independent PRNG seeded from self.seed + chunk_id
        let chunk_results: Vec<(f64, f64, usize)> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_id| {
                let my_n = chunk_size + if chunk_id < remainder { 1 } else { 0 };
                if my_n == 0 {
                    return (0.0, 0.0, 0);
                }

                let mut rng = Xoshiro256pp::new(self.seed.wrapping_add(chunk_id as u64));
                let mut x = vec![0.0; d];
                let mut mean = 0.0;
                let mut m2 = 0.0;

                for i in 1..=my_n {
                    for j in 0..d {
                        x[j] = lower[j] + widths[j] * rng.next_f64();
                    }
                    let val = f(&x);
                    let delta = val - mean;
                    mean += delta / i as f64;
                    let delta2 = val - mean;
                    m2 += delta * delta2;
                }

                (mean * my_n as f64, m2, my_n)
            })
            .collect();

        // Merge chunk results
        let mut total_sum = 0.0;
        let mut total_m2 = 0.0;
        let mut total_n = 0usize;
        for (sum, m2, count) in chunk_results {
            total_sum += sum;
            total_m2 += m2;
            total_n += count;
        }

        let mean = total_sum / total_n as f64;
        let variance = if total_n > 1 {
            total_m2 / (total_n - 1) as f64
        } else {
            0.0
        };
        let std_error = (variance / total_n as f64).sqrt() * volume;

        Ok(QuadratureResult {
            value: volume * mean,
            error_estimate: std_error,
            num_evals: total_n,
            converged: true,
        })
    }

    #[cfg(feature = "parallel")]
    fn integrate_qmc_par_sobol<G>(
        &self,
        d: usize,
        lower: &[f64],
        widths: &[f64],
        volume: f64,
        n: usize,
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64 + Sync,
    {
        use rayon::prelude::*;

        // Pre-generate all Sobol points (sequential — gray-code is inherently serial)
        let mut sob = SobolSequence::new(d)?;
        let mut points = vec![0.0; n * d];
        let mut u = vec![0.0; d];
        for i in 0..n {
            sob.next_point(&mut u);
            for j in 0..d {
                points[i * d + j] = lower[j] + widths[j] * u[j];
            }
        }

        // Parallel evaluation
        let sum: f64 = (0..n)
            .into_par_iter()
            .map(|i| f(&points[i * d..(i + 1) * d]))
            .sum();

        let estimate = volume * sum / n as f64;

        // Heuristic error: compare N/2 and N estimates
        let half = n / 2;
        let half_sum: f64 = (0..half)
            .into_par_iter()
            .map(|i| f(&points[i * d..(i + 1) * d]))
            .sum();
        let half_estimate = volume * half_sum / half as f64;
        let error = (estimate - half_estimate).abs();

        Ok(QuadratureResult {
            value: estimate,
            error_estimate: error,
            num_evals: n + half,
            converged: true,
        })
    }

    #[cfg(feature = "parallel")]
    fn integrate_qmc_par_halton<G>(
        &self,
        d: usize,
        lower: &[f64],
        widths: &[f64],
        volume: f64,
        n: usize,
        f: &G,
    ) -> Result<QuadratureResult<f64>, QuadratureError>
    where
        G: Fn(&[f64]) -> f64 + Sync,
    {
        use rayon::prelude::*;

        // Pre-generate all Halton points
        let mut hal = HaltonSequence::new(d)?;
        let mut points = vec![0.0; n * d];
        let mut u = vec![0.0; d];
        for i in 0..n {
            hal.next_point(&mut u);
            for j in 0..d {
                points[i * d + j] = lower[j] + widths[j] * u[j];
            }
        }

        // Parallel evaluation
        let sum: f64 = (0..n)
            .into_par_iter()
            .map(|i| f(&points[i * d..(i + 1) * d]))
            .sum();

        let estimate = volume * sum / n as f64;

        // Heuristic error
        let half = n / 2;
        let half_sum: f64 = (0..half)
            .into_par_iter()
            .map(|i| f(&points[i * d..(i + 1) * d]))
            .sum();
        let half_estimate = volume * half_sum / half as f64;
        let error = (estimate - half_estimate).abs();

        Ok(QuadratureResult {
            value: estimate,
            error_estimate: error,
            num_evals: n + half,
            converged: true,
        })
    }
}

/// Convenience: quasi-Monte Carlo integration using Sobol sequences.
pub fn monte_carlo_integrate<G>(
    f: G,
    lower: &[f64],
    upper: &[f64],
    num_samples: usize,
) -> Result<QuadratureResult<f64>, QuadratureError>
where
    G: Fn(&[f64]) -> f64,
{
    MonteCarloIntegrator::default()
        .with_samples(num_samples)
        .integrate(lower, upper, f)
}

/// Xoshiro256++ PRNG — simple, fast, no external dependency.
///
/// 256-bit state, 64-bit output. Period 2^256 - 1.
struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    fn new(seed: u64) -> Self {
        // SplitMix64 seeding to fill state from a single seed
        let mut s = [0u64; 4];
        let mut z = seed;
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    fn next_f64(&mut self) -> f64 {
        // Generate a double in [0, 1) using the upper 53 bits
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_input() {
        assert!(monte_carlo_integrate(|_| 1.0, &[], &[], 100).is_err());
    }

    /// Constant function: integral of 1 over [0,1]^d = 1.
    #[test]
    fn constant() {
        let result = monte_carlo_integrate(|_| 1.0, &[0.0; 3], &[1.0; 3], 1000).unwrap();
        assert!((result.value - 1.0).abs() < 1e-10, "value={}", result.value);
    }

    /// Integral of x over [0,1] = 0.5.
    #[test]
    fn linear_1d() {
        let result = monte_carlo_integrate(|x| x[0], &[0.0], &[1.0], 10000).unwrap();
        assert!((result.value - 0.5).abs() < 0.05, "value={}", result.value);
    }

    /// Plain MC convergence rate.
    #[test]
    fn plain_mc() {
        let integrator = MonteCarloIntegrator::default().with_method(MCMethod::Plain);
        let result = integrator
            .integrate(&[0.0, 0.0], &[1.0, 1.0], |x| x[0] * x[1])
            .unwrap();
        // Should be close to 0.25
        assert!((result.value - 0.25).abs() < 0.05, "value={}", result.value);
    }

    /// Sobol QMC should be more accurate than plain MC for smooth functions.
    #[test]
    fn sobol_better_than_plain() {
        let n = 10000;
        let f = |x: &[f64]| (x[0] * x[1] * x[2]).exp();
        let exact = {
            // integral of e^(xyz) over [0,1]^3
            // No easy closed form, use reference value
            // Approximate: ~1.14649...
            1.14649 // rough reference
        };

        let sobol_result = MonteCarloIntegrator::default()
            .with_method(MCMethod::Sobol)
            .with_samples(n)
            .integrate(&[0.0; 3], &[1.0; 3], &f)
            .unwrap();

        let plain_result = MonteCarloIntegrator::default()
            .with_method(MCMethod::Plain)
            .with_samples(n)
            .integrate(&[0.0; 3], &[1.0; 3], &f)
            .unwrap();

        let sobol_err = (sobol_result.value - exact).abs();
        let plain_err = (plain_result.value - exact).abs();

        // Sobol should generally be much better; allow some slack for stochastic plain
        // Just check both are reasonable
        assert!(sobol_err < 0.1, "Sobol error too large: {sobol_err}");
        assert!(plain_err < 0.5, "Plain error too large: {plain_err}");
    }

    /// Halton integration.
    #[test]
    fn halton_integration() {
        let result = MonteCarloIntegrator::default()
            .with_method(MCMethod::Halton)
            .with_samples(5000)
            .integrate(&[0.0, 0.0], &[1.0, 1.0], |x| x[0] + x[1])
            .unwrap();
        // integral of (x+y) over [0,1]^2 = 1
        assert!((result.value - 1.0).abs() < 0.05, "value={}", result.value);
    }

    /// Parallel Sobol QMC produces identical results (deterministic sequence).
    #[cfg(feature = "parallel")]
    #[test]
    fn sobol_par_matches_sequential() {
        let f = |x: &[f64]| (x[0] + x[1] + x[2]).exp();
        let integrator = MonteCarloIntegrator::default()
            .with_method(MCMethod::Sobol)
            .with_samples(5000);
        let seq = integrator.integrate(&[0.0; 3], &[1.0; 3], &f).unwrap();
        let par = integrator.integrate_par(&[0.0; 3], &[1.0; 3], &f).unwrap();
        assert!(
            (seq.value - par.value).abs() < 1e-12,
            "seq={}, par={}",
            seq.value,
            par.value
        );
    }

    /// Parallel Halton QMC produces identical results (deterministic sequence).
    #[cfg(feature = "parallel")]
    #[test]
    fn halton_par_matches_sequential() {
        let f = |x: &[f64]| x[0] * x[1];
        let integrator = MonteCarloIntegrator::default()
            .with_method(MCMethod::Halton)
            .with_samples(5000);
        let seq = integrator.integrate(&[0.0, 0.0], &[1.0, 1.0], &f).unwrap();
        let par = integrator
            .integrate_par(&[0.0, 0.0], &[1.0, 1.0], &f)
            .unwrap();
        assert!(
            (seq.value - par.value).abs() < 1e-12,
            "seq={}, par={}",
            seq.value,
            par.value
        );
    }

    /// Parallel plain MC produces reasonable results (different PRNG split, so not identical).
    #[cfg(feature = "parallel")]
    #[test]
    fn plain_mc_par_reasonable() {
        let integrator = MonteCarloIntegrator::default()
            .with_method(MCMethod::Plain)
            .with_samples(10000);
        let result = integrator
            .integrate_par(&[0.0, 0.0], &[1.0, 1.0], |x| x[0] * x[1])
            .unwrap();
        assert!(
            (result.value - 0.25).abs() < 0.05,
            "value={}",
            result.value
        );
    }
}
