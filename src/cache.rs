//! Precomputed quadrature rule cache.
//!
//! Provides lazily-initialised, globally-shared instances of common
//! Gauss-Legendre rules. Rules are computed once on first access and
//! reused for the lifetime of the process.
//!
//! # Example
//!
//! ```
//! use bilby::cache::GL10;
//!
//! let result = GL10.rule().integrate(0.0, 1.0, |x: f64| x * x);
//! assert!((result - 1.0 / 3.0).abs() < 1e-14);
//! ```

use std::sync::LazyLock;

use crate::gauss_legendre::GaussLegendre;

/// Precomputed 5-point Gauss-Legendre rule.
pub static GL5: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(5).unwrap());

/// Precomputed 10-point Gauss-Legendre rule.
pub static GL10: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(10).unwrap());

/// Precomputed 15-point Gauss-Legendre rule.
pub static GL15: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(15).unwrap());

/// Precomputed 20-point Gauss-Legendre rule.
pub static GL20: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(20).unwrap());

/// Precomputed 50-point Gauss-Legendre rule.
pub static GL50: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(50).unwrap());

/// Precomputed 100-point Gauss-Legendre rule.
pub static GL100: LazyLock<GaussLegendre> = LazyLock::new(|| GaussLegendre::new(100).unwrap());
