#![cfg_attr(not(feature = "std"), no_std)]
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
//! - **`no_std` compatible** — works without the standard library (with `alloc`)
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | Yes | Enables `std::error::Error` impl and [`cache`] module |
//! | `parallel` | No | Enables rayon-based `_par` methods (implies `std`) |
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
//! let (estimate, error) = gk.integrate(0.0, core::f64::consts::PI, f64::sin);
//! assert!((estimate - 2.0).abs() < 1e-14);
//! ```
//!
//! ## Adaptive Integration
//!
//! ```
//! use bilby::adaptive_integrate;
//!
//! let result = adaptive_integrate(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 1e-12).unwrap();
//! assert!((result.value - 2.0).abs() < 1e-12);
//! assert!(result.is_converged());
//! ```
//!
//! ## Infinite Domains
//!
//! ```
//! use bilby::integrate_infinite;
//!
//! // Integral of exp(-x^2) over (-inf, inf) = sqrt(pi)
//! let result = integrate_infinite(|x: f64| (-x * x).exp(), 1e-10).unwrap();
//! assert!((result.value - core::f64::consts::PI.sqrt()).abs() < 1e-8);
//! ```
//!
//! ## Multi-Dimensional Integration
//!
//! ```
//! use bilby::cubature::{TensorProductRule, adaptive_cubature};
//! use bilby::GaussLegendre;
//!
//! // Tensor product: 10-point GL in each of 2 dimensions
//! let gl = GaussLegendre::new(10).unwrap();
//! let tp = TensorProductRule::isotropic(gl.rule(), 2).unwrap();
//! let result = tp.rule().integrate_box(
//!     &[0.0, 0.0], &[1.0, 1.0],
//!     |x| x[0] * x[1],
//! );
//! assert!((result - 0.25).abs() < 1e-14);
//! ```
//!
//! ## Precomputed Rule Cache
//!
//! Available when the `std` feature is enabled (default):
//!
//! ```
//! # #[cfg(feature = "std")] {
//! use bilby::cache::GL10;
//!
//! let result = GL10.rule().integrate(0.0, 1.0, |x: f64| x * x);
//! assert!((result - 1.0 / 3.0).abs() < 1e-14);
//! # }
//! ```

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod adaptive;
#[cfg(feature = "std")]
pub mod cache;
pub mod cauchy_pv;
pub mod clenshaw_curtis;
pub mod cubature;
pub mod error;
pub mod gauss_chebyshev;
pub mod gauss_hermite;
pub mod gauss_jacobi;
pub mod gauss_kronrod;
pub mod gauss_laguerre;
pub mod gauss_legendre;
pub mod gauss_lobatto;
pub mod gauss_radau;
pub(crate) mod golub_welsch;
pub mod oscillatory;
pub mod result;
pub mod rule;
pub mod tanh_sinh;
pub mod transforms;
pub mod weighted;

pub use adaptive::{adaptive_integrate, adaptive_integrate_with_breaks, AdaptiveIntegrator};
pub use cauchy_pv::{pv_integrate, CauchyPV};
pub use clenshaw_curtis::ClenshawCurtis;
pub use error::QuadratureError;
pub use gauss_chebyshev::{GaussChebyshevFirstKind, GaussChebyshevSecondKind};
pub use gauss_hermite::GaussHermite;
pub use gauss_jacobi::GaussJacobi;
pub use gauss_kronrod::{GKPair, GaussKronrod};
pub use gauss_laguerre::GaussLaguerre;
pub use gauss_legendre::GaussLegendre;
pub use gauss_lobatto::GaussLobatto;
pub use gauss_radau::GaussRadau;
pub use oscillatory::{
    integrate_oscillatory_cos, integrate_oscillatory_sin, OscillatoryIntegrator, OscillatoryKernel,
};
pub use result::QuadratureResult;
pub use rule::QuadratureRule;
pub use tanh_sinh::{tanh_sinh_integrate, TanhSinh};
pub use transforms::{
    integrate_infinite, integrate_semi_infinite_lower, integrate_semi_infinite_upper,
};
pub use weighted::{weighted_integrate, WeightFunction, WeightedIntegrator};
