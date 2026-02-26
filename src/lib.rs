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
//! ```rust,ignore
//! use bilby::GaussLegendre;
//!
//! // Create a 10-point Gauss-Legendre rule
//! let gl = GaussLegendre::new(10);
//!
//! // Integrate x^2 over [0, 1]
//! let result = gl.integrate(0.0, 1.0, |x| x * x);
//! assert!((result - 1.0 / 3.0).abs() < 1e-14);
//! ```

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        // Tests will be added with the first rule implementation.
    }
}
