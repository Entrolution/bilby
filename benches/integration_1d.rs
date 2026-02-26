use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use bilby::{
    adaptive_integrate, tanh_sinh_integrate, GKPair, GaussKronrod, GaussLegendre,
    OscillatoryIntegrator, OscillatoryKernel,
};

fn bench_fixed_order_gl(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_gl_sin");
    for n in [10, 20, 50] {
        let gl = GaussLegendre::new(n).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &gl, |b, gl| {
            b.iter(|| {
                gl.rule()
                    .integrate(0.0, std::f64::consts::PI, |x: f64| x.sin())
            });
        });
    }
    group.finish();
}

fn bench_composite_gl(c: &mut Criterion) {
    let gl = GaussLegendre::new(10).unwrap();
    let mut group = c.benchmark_group("composite_gl_sin");
    for panels in [10, 100, 1_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(panels),
            &panels,
            |b, &panels| {
                b.iter(|| {
                    gl.rule().integrate_composite(
                        0.0,
                        std::f64::consts::PI,
                        panels,
                        |x: f64| x.sin(),
                    )
                });
            },
        );
    }
    group.finish();
}

fn bench_gk_pairs(c: &mut Criterion) {
    let mut group = c.benchmark_group("gk_sin");
    for (name, pair) in [("G7K15", GKPair::G7K15), ("G15K31", GKPair::G15K31)] {
        let gk = GaussKronrod::new(pair);
        group.bench_function(name, |b| {
            b.iter(|| gk.integrate(0.0, std::f64::consts::PI, |x: f64| x.sin()));
        });
    }
    group.finish();
}

fn bench_adaptive(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive");
    group.bench_function("smooth_sin", |b| {
        b.iter(|| {
            adaptive_integrate(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-10).unwrap()
        });
    });
    group.bench_function("peaked", |b| {
        b.iter(|| {
            adaptive_integrate(|x: f64| 1.0 / (1.0 + (x - 0.3).powi(2) * 1e4), 0.0, 1.0, 1e-8)
                .unwrap()
        });
    });
    group.bench_function("singular_sqrt", |b| {
        b.iter(|| {
            adaptive_integrate(|x: f64| 1.0 / x.sqrt(), 1e-15, 1.0, 1e-8).unwrap()
        });
    });
    group.finish();
}

fn bench_tanh_sinh(c: &mut Criterion) {
    let mut group = c.benchmark_group("tanh_sinh");
    group.bench_function("smooth_sin", |b| {
        b.iter(|| {
            tanh_sinh_integrate(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 1e-10).unwrap()
        });
    });
    group.bench_function("singular_sqrt", |b| {
        b.iter(|| tanh_sinh_integrate(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-8).unwrap());
    });
    group.finish();
}

fn bench_oscillatory(c: &mut Criterion) {
    let mut group = c.benchmark_group("oscillatory");
    for omega in [10.0, 100.0] {
        group.bench_with_input(
            BenchmarkId::new("sin", omega as u64),
            &omega,
            |b, &omega| {
                let integrator = OscillatoryIntegrator::new(OscillatoryKernel::Sine, omega)
                    .with_order(64)
                    .with_abs_tol(1e-10);
                b.iter(|| integrator.integrate(0.0, 1.0, |_| 1.0).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fixed_order_gl,
    bench_composite_gl,
    bench_gk_pairs,
    bench_adaptive,
    bench_tanh_sinh,
    bench_oscillatory,
);
criterion_main!(benches);
