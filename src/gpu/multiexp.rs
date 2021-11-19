use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

use ec_gpu_gen::error::EcResult;
use ec_gpu_gen::multiexp::MultiexpKernel;
use ec_gpu_gen::multiexp_cpu::{multiexp_cpu, FullDensity};
use ec_gpu_gen::threadpool::Worker;
use ff::PrimeField;
use group::{prime::PrimeCurveAffine, Group};
use log::{error, info};
use pairing::Engine;
use rust_gpu_tools::Device;

use crate::gpu::GpuEngine;

pub fn get_cpu_utilization() -> f64 {
    use std::env;
    env::var("BELLMAN_CPU_UTILIZATION")
        .map_or(0f64, |v| match v.parse() {
            Ok(val) => val,
            Err(_) => {
                error!("Invalid BELLMAN_CPU_UTILIZATION! Defaulting to 0...");
                0f64
            }
        })
        .max(0f64)
        .min(1f64)
}

// A Multiexp kernel that can share the workload between the GPU and the CPU.
pub struct CpuGpuMultiexpKernel<'a, E>(MultiexpKernel<'a, E>)
where
    E: Engine + GpuEngine;

impl<'a, E> CpuGpuMultiexpKernel<'a, E>
where
    E: Engine + GpuEngine,
{
    pub fn create(devices: &[&Device]) -> EcResult<Self> {
        info!("Multiexp: CPU utilization: {}.", get_cpu_utilization());
        let kernel = MultiexpKernel::create(devices)?;
        Ok(Self(kernel))
    }

    // TODO vmx 2021-11-16: document `maybe_abort`
    pub fn create_with_abort(
        devices: &[&Device],
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        info!("Multiexp: CPU utilization: {}.", get_cpu_utilization());
        let kernel = MultiexpKernel::create_with_abort(devices, maybe_abort)?;
        Ok(Self(kernel))
    }

    pub fn multiexp<G>(
        &mut self,
        pool: &Worker,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
        skip: usize,
    ) -> EcResult<<G as PrimeCurveAffine>::Curve>
    where
        G: PrimeCurveAffine<Scalar = E::Fr>,
    {
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases[skip..(skip + exps.len())];
        let exps = &exps[..];

        let cpu_n = ((exps.len() as f64) * get_cpu_utilization()) as usize;
        let n = exps.len() - cpu_n;
        let (cpu_bases, bases) = bases.split_at(cpu_n);
        let (cpu_exps, exps) = exps.split_at(cpu_n);

        let mut results = Vec::new();
        let error = Arc::new(RwLock::new(Ok(())));

        let cpu_acc = pool.scoped(|s| {
            if n > 0 {
                results = vec![<G as PrimeCurveAffine>::Curve::identity(); self.0.num_kernels()];
                self.0
                    .parallel_multiexp(s, bases, exps, &mut results, error.clone());
            }

            multiexp_cpu::<_, _, _, E, _>(
                &pool,
                (Arc::new(cpu_bases.to_vec()), 0),
                FullDensity,
                Arc::new(cpu_exps.to_vec()),
            )
        });

        Arc::try_unwrap(error)
            .expect("only one ref left")
            .into_inner()
            .unwrap()?;
        let mut acc = <G as PrimeCurveAffine>::Curve::identity();
        for r in results {
            acc.add_assign(&r);
        }

        acc.add_assign(&cpu_acc.wait().unwrap());
        Ok(acc)
    }
}
