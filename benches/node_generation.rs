use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use bilby::{ClenshawCurtis, GaussHermite, GaussJacobi, GaussLaguerre, GaussLegendre};

fn bench_gl_newton(c: &mut Criterion) {
    let mut group = c.benchmark_group("gl_newton");
    for n in [10, 50, 100] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| GaussLegendre::new(n).unwrap());
        });
    }
    group.finish();
}

fn bench_gl_bogaert(c: &mut Criterion) {
    let mut group = c.benchmark_group("gl_bogaert");
    for n in [200, 1_000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| GaussLegendre::new(n).unwrap());
        });
    }
    group.finish();
}

fn bench_golub_welsch(c: &mut Criterion) {
    let mut group = c.benchmark_group("golub_welsch");
    group.bench_function("jacobi_n50", |b| {
        b.iter(|| GaussJacobi::new(50, 0.5, 0.5).unwrap());
    });
    group.bench_function("hermite_n50", |b| {
        b.iter(|| GaussHermite::new(50).unwrap());
    });
    group.bench_function("laguerre_n50", |b| {
        b.iter(|| GaussLaguerre::new(50, 0.0).unwrap());
    });
    group.finish();
}

fn bench_clenshaw_curtis(c: &mut Criterion) {
    let mut group = c.benchmark_group("clenshaw_curtis");
    for n in [33, 65, 129] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| ClenshawCurtis::new(n).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gl_newton,
    bench_gl_bogaert,
    bench_golub_welsch,
    bench_clenshaw_curtis,
);
criterion_main!(benches);
