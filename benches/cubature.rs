use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use bilby::cubature::{
    adaptive_cubature, monte_carlo_integrate, MCMethod, MonteCarloIntegrator, SparseGrid,
    TensorProductRule,
};
use bilby::GaussLegendre;

fn bench_tensor_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_product");

    // d=2, n=10
    let gl10 = GaussLegendre::new(10).unwrap();
    let tp_2d = TensorProductRule::isotropic(gl10.rule(), 2).unwrap();
    group.bench_function("d2_n10", |b| {
        b.iter(|| {
            tp_2d
                .rule()
                .integrate_box(&[0.0, 0.0], &[1.0, 1.0], |x| x[0] * x[1])
        });
    });

    // d=3, n=5
    let gl5 = GaussLegendre::new(5).unwrap();
    let tp_3d = TensorProductRule::isotropic(gl5.rule(), 3).unwrap();
    group.bench_function("d3_n5", |b| {
        b.iter(|| {
            tp_3d
                .rule()
                .integrate_box(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0], |x| x[0] * x[1] * x[2])
        });
    });

    group.finish();
}

fn bench_sparse_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_grid");

    let sg_3_3 = SparseGrid::clenshaw_curtis(3, 3).unwrap();
    group.bench_function("d3_l3", |b| {
        b.iter(|| {
            sg_3_3
                .rule()
                .integrate_box(&[0.0; 3], &[1.0; 3], |x| (x[0] + x[1] + x[2]).exp())
        });
    });

    let sg_5_3 = SparseGrid::clenshaw_curtis(5, 3).unwrap();
    group.bench_function("d5_l3", |b| {
        b.iter(|| {
            sg_5_3.rule().integrate_box(&[0.0; 5], &[1.0; 5], |x| {
                (x[0] + x[1] + x[2] + x[3] + x[4]).exp()
            })
        });
    });

    group.finish();
}

fn bench_adaptive_cubature(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_cubature");

    group.bench_function("d2_gaussian", |b| {
        b.iter(|| {
            adaptive_cubature(
                |x| (-x[0] * x[0] - x[1] * x[1]).exp(),
                &[0.0, 0.0],
                &[1.0, 1.0],
                1e-6,
            )
            .unwrap()
        });
    });

    group.bench_function("d3_polynomial", |b| {
        b.iter(|| {
            adaptive_cubature(
                |x| x[0] * x[0] + x[1] * x[1] + x[2] * x[2],
                &[0.0, 0.0, 0.0],
                &[1.0, 1.0, 1.0],
                1e-8,
            )
            .unwrap()
        });
    });

    group.finish();
}

fn bench_monte_carlo(c: &mut Criterion) {
    let mut group = c.benchmark_group("monte_carlo");

    let f = |x: &[f64]| (x[0] * x[1] * x[2]).exp();
    let lower = [0.0; 3];
    let upper = [1.0; 3];
    let n = 10_000;

    for (name, method) in [
        ("plain", MCMethod::Plain),
        ("sobol", MCMethod::Sobol),
        ("halton", MCMethod::Halton),
    ] {
        group.bench_with_input(BenchmarkId::new("3d_exp", name), &method, |b, &method| {
            b.iter(|| {
                MonteCarloIntegrator::default()
                    .with_method(method)
                    .with_samples(n)
                    .integrate(&lower, &upper, &f)
                    .unwrap()
            });
        });
    }

    // Also benchmark the convenience function
    group.bench_function("sobol_convenience_10k", |b| {
        b.iter(|| monte_carlo_integrate(&f, &lower, &upper, n).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_product,
    bench_sparse_grid,
    bench_adaptive_cubature,
    bench_monte_carlo,
);
criterion_main!(benches);
