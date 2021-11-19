use std::sync::Arc;

use ec_gpu_gen::error::EcError;
#[cfg(any(feature = "cuda", feature = "opencl"))]
use ec_gpu_gen::multiexp::MultiexpKernel;
use ec_gpu_gen::multiexp_cpu::{multiexp_cpu, QueryDensity, SourceBuilder};
use ec_gpu_gen::threadpool::{Waiter, Worker};
use ff::PrimeField;
use group::prime::PrimeCurveAffine;
use pairing::Engine;

use crate::gpu;

/// Perform multi-exponentiation. The caller is responsible for ensuring the
/// query size is the same as the number of exponents.
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn multiexp<'b, Q, D, G, E, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    kern: &mut gpu::LockedMultiexpKernel<E>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine,
    E: gpu::GpuEngine,
    E: Engine<Fr = G::Scalar>,
    S: SourceBuilder<G>,
{
    // Try to run on the GPU.
    if let Ok(p) = kern.with(|k: &mut MultiexpKernel<E>| {
        let exps = density_map.as_ref().generate_exps::<E>(exponents.clone());
        let (bss, skip) = bases.clone().get();
        k.multiexp(pool, bss, exps, skip).map_err(Into::into)
    }) {
        return Waiter::done(Ok(p));
    }

    // Fallback to the CPU in case the GPU run failed.
    let result_cpu = multiexp_cpu::<_, _, _, E, _>(pool, bases, density_map, exponents);

    // Do not give the control back to the caller till the multiexp is done. Once done the GPU
    // might again be free, so we can run subsequent calls on the GPU instead of the CPU again.
    let result = result_cpu.wait();

    Waiter::done(result)
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
pub fn multiexp<'b, Q, D, G, E, S>(
    pool: &Worker,
    bases: S,
    density_map: D,
    exponents: Arc<Vec<<G::Scalar as PrimeField>::Repr>>,
    _kern: &mut gpu::LockedMultiexpKernel<E>,
) -> Waiter<Result<<G as PrimeCurveAffine>::Curve, EcError>>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
    G: PrimeCurveAffine,
    E: gpu::GpuEngine,
    E: Engine<Fr = G::Scalar>,
    S: SourceBuilder<G>,
{
    multiexp_cpu::<_, _, _, E, _>(pool, bases, density_map, exponents)
}

//#[test]
//fn test_with_bls12() {
//    fn naive_multiexp<G: PrimeCurveAffine>(
//        bases: Arc<Vec<G>>,
//        exponents: &[G::Scalar],
//    ) -> G::Curve {
//        assert_eq!(bases.len(), exponents.len());
//
//        let mut acc = G::Curve::identity();
//
//        for (base, exp) in bases.iter().zip(exponents.iter()) {
//            acc.add_assign(&base.mul(*exp));
//        }
//
//        acc
//    }
//
//    use blstrs::Bls12;
//    use group::Curve;
//
//    const SAMPLES: usize = 1 << 14;
//
//    let rng = &mut rand::thread_rng();
//    let v: Vec<<Bls12 as Engine>::Fr> = (0..SAMPLES)
//        .map(|_| <Bls12 as Engine>::Fr::random(&mut *rng))
//        .collect();
//    let g = Arc::new(
//        (0..SAMPLES)
//            .map(|_| <Bls12 as Engine>::G1::random(&mut *rng).to_affine())
//            .collect::<Vec<_>>(),
//    );
//
//    let now = std::time::Instant::now();
//    let naive = naive_multiexp(g.clone(), &v);
//    println!("Naive: {}", now.elapsed().as_millis());
//
//    let now = std::time::Instant::now();
//    let pool = Worker::new();
//
//    let v = Arc::new(v.into_iter().map(|fr| fr.to_repr()).collect());
//    let fast = multiexp_cpu::<_, _, _, Bls12, _>(&pool, (g, 0), FullDensity, v)
//        .wait()
//        .unwrap();
//
//    println!("Fast: {}", now.elapsed().as_millis());
//
//    assert_eq!(naive, fast);
//}
//
//#[cfg(any(feature = "cuda", feature = "opencl"))]
//#[test]
//pub fn gpu_multiexp_consistency() {
//    use blstrs::Bls12;
//    use group::Curve;
//    use std::time::Instant;
//
//    let _ = env_logger::try_init();
//
//    const MAX_LOG_D: usize = 16;
//    const START_LOG_D: usize = 10;
//    let mut kern = gpu::LockedMultiexpKernel::<Bls12>::new(MAX_LOG_D, false);
//    let pool = Worker::new();
//
//    let mut rng = rand::thread_rng();
//
//    let mut bases = (0..(1 << 10))
//        .map(|_| <Bls12 as Engine>::G1::random(&mut rng).to_affine())
//        .collect::<Vec<_>>();
//
//    for log_d in START_LOG_D..=MAX_LOG_D {
//        let g = Arc::new(bases.clone());
//
//        let samples = 1 << log_d;
//        println!("Testing Multiexp for {} elements...", samples);
//
//        let v = Arc::new(
//            (0..samples)
//                .map(|_| <Bls12 as Engine>::Fr::random(&mut rng).to_repr())
//                .collect::<Vec<_>>(),
//        );
//
//        let mut now = Instant::now();
//        let gpu = multiexp(&pool, (g.clone(), 0), FullDensity, v.clone(), &mut kern)
//            .wait()
//            .unwrap();
//        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
//        println!("GPU took {}ms.", gpu_dur);
//
//        now = Instant::now();
//        let cpu = multiexp_cpu::<_, _, _, Bls12, _>(&pool, (g.clone(), 0), FullDensity, v.clone())
//            .wait()
//            .unwrap();
//        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
//        println!("CPU took {}ms.", cpu_dur);
//
//        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);
//
//        assert_eq!(cpu, gpu);
//
//        println!("============================");
//
//        bases = [bases.clone(), bases.clone()].concat();
//    }
//}
//
//#[cfg(test)]
//mod tests {
//    use super::*;
//
//    use rand::Rng;
//    use rand_core::SeedableRng;
//    use rand_xorshift::XorShiftRng;
//
//    #[test]
//    fn test_extend_density_regular() {
//        let mut rng = XorShiftRng::from_seed([
//            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
//            0xbc, 0xe5,
//        ]);
//
//        for k in &[2, 4, 8] {
//            for j in &[10, 20, 50] {
//                let count: usize = k * j;
//
//                let mut tracker_full = DensityTracker::new();
//                let mut partial_trackers: Vec<DensityTracker> = Vec::with_capacity(count / k);
//                for i in 0..count {
//                    if i % k == 0 {
//                        partial_trackers.push(DensityTracker::new());
//                    }
//
//                    let index: usize = i / k;
//                    if rng.gen() {
//                        tracker_full.add_element();
//                        partial_trackers[index].add_element();
//                    }
//
//                    if !partial_trackers[index].bv.is_empty() {
//                        let idx = rng.gen_range(0..partial_trackers[index].bv.len());
//                        let offset: usize = partial_trackers
//                            .iter()
//                            .take(index)
//                            .map(|t| t.bv.len())
//                            .sum();
//                        tracker_full.inc(offset + idx);
//                        partial_trackers[index].inc(idx);
//                    }
//                }
//
//                let mut tracker_combined = DensityTracker::new();
//                for tracker in partial_trackers.into_iter() {
//                    tracker_combined.extend(tracker, false);
//                }
//                assert_eq!(tracker_combined, tracker_full);
//            }
//        }
//    }
//
//    #[test]
//    fn test_extend_density_input() {
//        let mut rng = XorShiftRng::from_seed([
//            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
//            0xbc, 0xe5,
//        ]);
//        let trials = 10;
//        let max_bits = 10;
//        let max_density = max_bits;
//
//        // Create an empty DensityTracker.
//        let empty = || DensityTracker::new();
//
//        // Create a random DensityTracker with first bit unset.
//        let unset = |rng: &mut XorShiftRng| {
//            let mut dt = DensityTracker::new();
//            dt.add_element();
//            let n = rng.gen_range(1..max_bits);
//            let target_density = rng.gen_range(0..max_density);
//            for _ in 1..n {
//                dt.add_element();
//            }
//
//            for _ in 0..target_density {
//                if n > 1 {
//                    let to_inc = rng.gen_range(1..n);
//                    dt.inc(to_inc);
//                }
//            }
//            assert!(!dt.bv[0]);
//            assert_eq!(n, dt.bv.len());
//            dbg!(&target_density, &dt.total_density);
//
//            dt
//        };
//
//        // Create a random DensityTracker with first bit set.
//        let set = |mut rng: &mut XorShiftRng| {
//            let mut dt = unset(&mut rng);
//            dt.inc(0);
//            dt
//        };
//
//        for _ in 0..trials {
//            {
//                // Both empty.
//                let (mut e1, e2) = (empty(), empty());
//                e1.extend(e2, true);
//                assert_eq!(empty(), e1);
//            }
//            {
//                // First empty, second unset.
//                let (mut e1, u1) = (empty(), unset(&mut rng));
//                e1.extend(u1.clone(), true);
//                assert_eq!(u1, e1);
//            }
//            {
//                // First empty, second set.
//                let (mut e1, s1) = (empty(), set(&mut rng));
//                e1.extend(s1.clone(), true);
//                assert_eq!(s1, e1);
//            }
//            {
//                // First set, second empty.
//                let (mut s1, e1) = (set(&mut rng), empty());
//                let s2 = s1.clone();
//                s1.extend(e1, true);
//                assert_eq!(s1, s2);
//            }
//            {
//                // First unset, second empty.
//                let (mut u1, e1) = (unset(&mut rng), empty());
//                let u2 = u1.clone();
//                u1.extend(e1, true);
//                assert_eq!(u1, u2);
//            }
//            {
//                // First unset, second unset.
//                let (mut u1, u2) = (unset(&mut rng), unset(&mut rng));
//                let expected_total = u1.total_density + u2.total_density;
//                u1.extend(u2, true);
//                assert_eq!(expected_total, u1.total_density);
//                assert!(!u1.bv[0]);
//            }
//            {
//                // First unset, second set.
//                let (mut u1, s1) = (unset(&mut rng), set(&mut rng));
//                let expected_total = u1.total_density + s1.total_density;
//                u1.extend(s1, true);
//                assert_eq!(expected_total, u1.total_density);
//                assert!(u1.bv[0]);
//            }
//            {
//                // First set, second unset.
//                let (mut s1, u1) = (set(&mut rng), unset(&mut rng));
//                let expected_total = s1.total_density + u1.total_density;
//                s1.extend(u1, true);
//                assert_eq!(expected_total, s1.total_density);
//                assert!(s1.bv[0]);
//            }
//            {
//                // First set, second set.
//                let (mut s1, s2) = (set(&mut rng), set(&mut rng));
//                let expected_total = s1.total_density + s2.total_density - 1;
//                s1.extend(s2, true);
//                assert_eq!(expected_total, s1.total_density);
//                assert!(s1.bv[0]);
//            }
//        }
//    }
//}
