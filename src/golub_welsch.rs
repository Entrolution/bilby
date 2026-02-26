//! Golub-Welsch algorithm for computing Gaussian quadrature rules
//! from three-term recurrence coefficients.
//!
//! Given the monic recurrence: x `p_k` = `p_{k+1}` + `α_k` `p_k` + `β_k` `p_{k-1}`,
//! the Jacobi matrix has diagonal `α_k` and off-diagonal √`β_{k+1}`.
//! Eigenvalues = quadrature nodes, weights = μ₀ · `z_k²` where `z_k` is the
//! first component of the k-th eigenvector and μ₀ is the zeroth moment.
//!
//! Reference: Golub & Welsch (1969), "Calculation of Gauss Quadrature Rules",
//! Mathematics of Computation 23(106), 221-230.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Compute quadrature nodes and weights from three-term recurrence coefficients.
///
/// # Arguments
/// - `diag`: diagonal elements α₀, ..., α_{n-1} of the Jacobi matrix
/// - `off_diag_sq`: squared off-diagonal elements β₁, ..., β_{n-1}
/// - `mu0`: zeroth moment ∫ w(x) dx of the weight function
///
/// # Returns
/// (nodes, weights) sorted ascending by node.
pub(crate) fn golub_welsch(diag: &[f64], off_diag_sq: &[f64], mu0: f64) -> (Vec<f64>, Vec<f64>) {
    let n = diag.len();
    assert_eq!(off_diag_sq.len(), n.saturating_sub(1));

    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![diag[0]], vec![mu0]);
    }

    let mut d = diag.to_vec();
    let mut e: Vec<f64> = off_diag_sq.iter().map(|&b| b.sqrt()).collect();

    // z[k] tracks the first component of the k-th eigenvector.
    // Initialized to e_0 = [1, 0, ..., 0] (first row of identity).
    let mut z = vec![0.0; n];
    z[0] = 1.0;

    symmetric_tridiag_eig(&mut d, &mut e, &mut z);

    let weights: Vec<f64> = z.iter().map(|&zk| mu0 * zk * zk).collect();

    // Sort ascending by node
    let mut pairs: Vec<_> = d.into_iter().zip(weights).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    pairs.into_iter().unzip()
}

/// Modify the last diagonal element of a Jacobi matrix so that a prescribed
/// node `x0` becomes an eigenvalue. Used for Gauss-Radau rules.
///
/// # Arguments
/// - `diag`: diagonal elements α₀, ..., α_{n-1} (modified in place)
/// - `off_diag_sq`: squared off-diagonal elements β₁, ..., β_{n-1}
/// - `x0`: the prescribed node (endpoint)
pub(crate) fn radau_modify(diag: &mut [f64], off_diag_sq: &[f64], x0: f64) {
    let n = diag.len();
    assert_eq!(off_diag_sq.len(), n.saturating_sub(1));

    if n <= 1 {
        diag[0] = x0;
        return;
    }

    // Evaluate the continued fraction r = p_{n-1}(x0) / p_{n-2}(x0)
    // using the characteristic polynomial recurrence:
    //   p_0 = 1
    //   p_1 = x0 - α_0
    //   p_k = (x0 - α_{k-1}) p_{k-1} - β_{k-1} p_{k-2}
    //
    // The ratio r_k = p_k / p_{k-1} satisfies:
    //   r_1 = x0 - α_0
    //   r_k = x0 - α_{k-1} - off_diag_sq[k-2] / r_{k-1}   for k >= 2
    let mut r = x0 - diag[0]; // r_1
    for k in 2..n {
        r = x0 - diag[k - 1] - off_diag_sq[k - 2] / r;
    }
    // Now r = r_{n-1} = p_{n-1}(x0) / p_{n-2}(x0)
    // Choose α_{n-1} so that p_n(x0) = 0:
    //   p_n = (x0 - α_{n-1}) p_{n-1} - β_{n-1} p_{n-2} = 0
    //   α_{n-1} = x0 - β_{n-1} / r
    diag[n - 1] = x0 - off_diag_sq[n - 2] / r;
}

/// Symmetric tridiagonal eigenvalue decomposition via the implicit QL
/// algorithm with Wilkinson shifts.
///
/// On entry: `d` = diagonal, `e` = off-diagonal (length n-1).
/// On exit: `d` = eigenvalues (unsorted), `z` = first row of eigenvector matrix.
#[allow(clippy::many_single_char_names)] // d, e, z, m, l, g, r, s, c, p, f, b are standard names in tridiagonal eigenvalue algorithms
fn symmetric_tridiag_eig(d: &mut [f64], e: &mut [f64], z: &mut [f64]) {
    let n = d.len();
    if n <= 1 {
        return;
    }

    // Pad e with a trailing zero for convenience
    let mut e_ext = vec![0.0; n];
    e_ext[..n - 1].copy_from_slice(e);

    for l in 0..n {
        let mut iter_count = 0u32;
        loop {
            // Find smallest m >= l such that e_ext[m] is negligible
            let mut m = l;
            while m < n - 1 {
                let tst = d[m].abs() + d[m + 1].abs();
                if e_ext[m].abs() <= f64::EPSILON * tst {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }

            iter_count += 1;
            if iter_count > 200 {
                break;
            }

            // Wilkinson shift
            let mut g = (d[l + 1] - d[l]) / (2.0 * e_ext[l]);
            let r = g.hypot(1.0);
            g = d[m] - d[l] + e_ext[l] / (g + r.copysign(g));

            let mut s = 1.0;
            let mut c = 1.0;
            let mut p = 0.0;
            let mut deflated = false;

            for i in (l..m).rev() {
                let f = s * e_ext[i];
                let b = c * e_ext[i];
                let r = f.hypot(g);
                e_ext[i + 1] = r;

                if r.abs() < 1e-300 {
                    // Near-zero: deflate
                    d[i + 1] -= p;
                    e_ext[m] = 0.0;
                    deflated = true;
                    break;
                }

                s = f / r;
                c = g / r;
                let g_tmp = d[i + 1] - p;
                let r2 = (d[i] - g_tmp) * s + 2.0 * c * b;
                p = s * r2;
                d[i + 1] = g_tmp + p;
                g = c * r2 - b;

                // Update first row of eigenvector matrix
                let fz = z[i + 1];
                z[i + 1] = s * z[i] + c * fz;
                z[i] = c * z[i] - s * fz;
            }

            if !deflated {
                d[l] -= p;
                e_ext[l] = g;
                e_ext[m] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golub-Welsch with Legendre recurrence should recover Gauss-Legendre.
    #[test]
    fn legendre_recovery() {
        let n = 5;
        let diag = vec![0.0; n]; // α_k = 0 for Legendre
        let off_diag_sq: Vec<f64> = (1..n)
            .map(|k| {
                let k = k as f64;
                k * k / (4.0 * k * k - 1.0)
            })
            .collect();
        let mu0 = 2.0;

        let (nodes, weights) = golub_welsch(&diag, &off_diag_sq, mu0);

        // Weight sum should be 2
        let sum: f64 = weights.iter().sum();
        assert!((sum - 2.0).abs() < 1e-14, "sum={sum}");

        // Nodes should be in (-1, 1) and sorted
        assert!(nodes[0] > -1.0);
        assert!(*nodes.last().unwrap() < 1.0);
        for i in 0..n - 1 {
            assert!(nodes[i] < nodes[i + 1]);
        }

        // Polynomial exactness: integral of x^4 over [-1,1] = 2/5
        let r: f64 = nodes
            .iter()
            .zip(&weights)
            .map(|(&x, &w)| x.powi(4) * w)
            .sum();
        assert!((r - 2.0 / 5.0).abs() < 1e-14, "r={r}");
    }

    /// Radau modification should produce a node at -1.
    #[test]
    fn radau_left_n2() {
        let n = 2;
        let mut diag = vec![0.0; n];
        let off_diag_sq = vec![1.0 / 3.0]; // β_1 = 1/3 for Legendre
        let mu0 = 2.0;

        radau_modify(&mut diag, &off_diag_sq, -1.0);
        let (nodes, weights) = golub_welsch(&diag, &off_diag_sq, mu0);

        assert!((nodes[0] - (-1.0)).abs() < 1e-14);
        assert!((nodes[1] - 1.0 / 3.0).abs() < 1e-14);
        assert!((weights[0] - 0.5).abs() < 1e-14, "w0={}", weights[0]);
        assert!((weights[1] - 1.5).abs() < 1e-14, "w1={}", weights[1]);
    }
}
