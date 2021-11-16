use ec_gpu::GpuEngine;
use ec_gpu_gen::Limb;

// WARNING: This function works only with Short Weierstrass Jacobian curves with Fq2 extension field.
pub fn kernel<E, L>() -> String
where
    E: GpuEngine,
    L: Limb,
{
    [
        ec_gpu_gen::gen_source::<E, L>(),
    ]
    .join("\n\n")
}
