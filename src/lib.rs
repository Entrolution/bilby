//! # bilby
//!
//! A high-performance numerical quadrature (integration) library for Rust.
//!
//! bilby provides Gaussian quadrature rules, adaptive integration, and
//! multi-dimensional cubature methods with a consistent, generic API.
//!
//! ## Design
//!
//! - **Generic over `F: Float`** via [`num_traits::Float`] — works with `f32`, `f64`,
//!   and compatible types (e.g., echidna AD types)
//! - **Precomputed rules** — generate nodes and weights once, integrate many times
//! - **No heap allocation on hot paths** — rule construction allocates, integration does not
//! - **Separate 1D and N-D APIs** — `Fn(F) -> F` for 1D, `Fn(&[F]) -> F` for N-D
//!
//! ## Quick Start
//!
//! ```
//! use bilby::GaussLegendre;
//!
//! // Create a 10-point Gauss-Legendre rule
//! let gl = GaussLegendre::new(10).unwrap();
//!
//! // Integrate x^2 over [0, 1] (exact result = 1/3)
//! let result = gl.rule().integrate(0.0, 1.0, |x: f64| x * x);
//! assert!((result - 1.0 / 3.0).abs() < 1e-14);
//! ```
//!
//! ## Gauss-Kronrod Error Estimation
//!
//! ```
//! use bilby::{GaussKronrod, GKPair};
//!
//! let gk = GaussKronrod::new(GKPair::G7K15);
//! let (estimate, error) = gk.integrate(0.0, std::f64::consts::PI, f64::sin);
//! assert!((estimate - 2.0).abs() < 1e-14);
//! ```

pub mod error;
pub mod gauss_kronrod;
pub mod gauss_legendre;
pub mod rule;

pub use error::QuadratureError;
pub use gauss_kronrod::{GKPair, GaussKronrod};
pub use gauss_legendre::GaussLegendre;
pub use rule::QuadratureRule;
