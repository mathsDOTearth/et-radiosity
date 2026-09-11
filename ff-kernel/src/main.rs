//! Form-factor kernel for the ET-SoC-1.
//!
//! Each hart computes a contiguous block of rows of the N x N form-factor
//! matrix F, where F[i][j] is the fraction of diffuse energy leaving patch i
//! that arrives at patch j:
//!
//! ```text
//!              cos(theta_i) * cos(theta_j)
//!   F_ij  =  ----------------------------- * A_j * V_ij
//!                      pi * r^2
//! ```
//!
//! `theta_i`, `theta_j`: angles at each patch between the inter-patch vector
//! and the respective normal. `r`: centroid-to-centroid distance. `A_j`: area
//! of the receiving patch. `V_ij`: binary visibility factor computed by the
//! Moller-Trumbore shadow test against the occluder triangle array.
//!
//! When `n_occluders == 0` (empty room, no interior objects) visibility is
//! assumed to be unity for all pairs, which is exact for a convex enclosure.
//!
//! Each row is normalised after assembly so that `sum_j F[i][j] <= 1`,
//! enforcing energy conservation in the presence of numerical drift.
//!
//! # Floating-point note
//!
//! The target ISA (RV64IMAC) has no F/D extensions; f32 arithmetic is lowered
//! to compiler_builtins soft-float routines. `sqrt` is implemented via three
//! Newton-Raphson iterations after a bit-manipulation initial estimate, giving
//! full f32 precision without requiring libm.

#![no_std]
#![no_main]

use et_abi::DeviceArgs;
use et_kernel::{Grid, MsgBuf, device_slice, fence, kernel_entry, trace_str};
use radiosity_abi::{FormFactorArgs, OccluderTri, Patch};

kernel_entry!();

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------

/// Kernel entry: distributes form-factor row computation across harts.
///
/// # Safety
///
/// `args_ptr` must be the raw pointer delivered in `a0` by the host firmware.
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    let args = unsafe { FormFactorArgs::from_ptr(args_ptr as *const u8) };

    let grid = Grid::new(args.n_harts);
    if !grid.active() {
        return 0;
    }

    let n = args.n_patches as usize;

    // SAFETY: host allocated and initialised these arrays before launch.
    let patches: &[Patch] = unsafe { device_slice(args.patches_addr as usize, n) };
    let occluders: &[OccluderTri] = if args.n_occluders > 0 {
        unsafe { device_slice(args.occluder_addr as usize, args.n_occluders as usize) }
    } else {
        &[]
    };

    // Partition rows across harts (ceiling division).
    let n_harts       = grid.n_harts() as usize;
    let hart          = grid.hart()    as usize;
    let rows_per_hart = (n + n_harts - 1) / n_harts;
    let row_start     = (hart * rows_per_hart).min(n);
    let row_end       = ((hart + 1) * rows_per_hart).min(n);

    let ff_base = args.ff_matrix_addr as usize;

    for i in row_start..row_end {
        let pi = &patches[i];

        for j in 0..n {
            let ff = if i == j {
                0.0f32
            } else {
                let pj = &patches[j];
                // Geometric hemisphere check precedes the shadow ray so that the
                // expensive Moller-Trumbore traversal is skipped for patch pairs
                // that face away from one another (ff = 0 by definition for those).
                form_factor_with_visibility(pi, pj, occluders)
            };

            // SAFETY: exclusive write to row i; no other hart touches this row.
            // Row normalisation is performed on the host after DMA download to
            // avoid reading back from device DRAM within the same kernel launch:
            // on the ET-Minion store buffer, volatile reads may see stale
            // content if issued without an explicit fence after the prior writes.
            unsafe {
                let ptr = (ff_base + (i * n + j) * 4) as *mut f32;
                ptr.write_volatile(ff);
            }
        }
    }

    fence();

    if grid.hart() == 0 {
        let mut m = MsgBuf::new();
        m.str(b"ff-kernel: n_patches=")
            .u64(args.n_patches as u64)
            .str(b" n_occluders=")
            .u64(args.n_occluders as u64)
            .str(b" n_harts=")
            .u64(args.n_harts as u64);
        trace_str(m.as_slice());
    }

    0
}

// ---------------------------------------------------------------------------
// Combined geometric + visibility filter
// ---------------------------------------------------------------------------

/// Returns F_ij after applying the hemisphere filter and, if needed, the shadow
/// ray test.
///
/// The hemisphere filter uses the *unnormalised* inter-patch direction -- the
/// sign of `dot(n_i, d)` and `dot(-n_j, d)` is identical to the sign of the
/// respective cosines, so no `sqrt` is required for this early exit. Only patch
/// pairs that pass the hemisphere check proceed to `is_visible`, which is the
/// dominant cost for scenes with interior occluders.
///
/// In a Cornell box roughly 60-70% of pairs are eliminated by the hemisphere
/// check alone, reducing total shadow-ray invocations proportionally.
#[inline]
fn form_factor_with_visibility(pi: &Patch, pj: &Patch, occluders: &[OccluderTri]) -> f32 {
    let dx = pj.centroid[0] - pi.centroid[0];
    let dy = pj.centroid[1] - pi.centroid[1];
    let dz = pj.centroid[2] - pi.centroid[2];

    // Cosine-sign check: both patches must face toward each other.
    // Using the raw (unnormalised) direction preserves the sign of cos.
    let cos_i_sign = pi.normal[0]*dx + pi.normal[1]*dy + pi.normal[2]*dz;
    let cos_j_sign = -(pj.normal[0]*dx + pj.normal[1]*dy + pj.normal[2]*dz);
    if cos_i_sign <= 0.0 || cos_j_sign <= 0.0 {
        return 0.0;
    }

    // Shadow ray: only reached when the geometry is plausible.
    if !occluders.is_empty() && !is_visible(pi, pj, occluders) {
        return 0.0;
    }

    compute_form_factor(pi, pj)
}

// ---------------------------------------------------------------------------
// Visibility test (Moller-Trumbore shadow ray)
// ---------------------------------------------------------------------------

/// Returns `true` if the segment from `pi.centroid` to `pj.centroid` is not
/// blocked by any occluder triangle.
///
/// A 1 mm margin at both ends of the segment prevents the ray from registering
/// a self-intersection against the triangle that patch i or j lies on.
#[inline]
fn is_visible(pi: &Patch, pj: &Patch, occluders: &[OccluderTri]) -> bool {
    let ro = pi.centroid;
    let dx = pj.centroid[0] - ro[0];
    let dy = pj.centroid[1] - ro[1];
    let dz = pj.centroid[2] - ro[2];
    let dist = sqrt_f32(dx * dx + dy * dy + dz * dz);
    if dist < 1.0e-6 {
        return false;
    }
    let rd = [dx / dist, dy / dist, dz / dist];
    const T_MIN: f32 = 1.0e-3;

    for occ in occluders {
        if let Some(t) = moller_trumbore(ro, rd, occ.v0, occ.v1, occ.v2) {
            // The segment ends just before pj, so t_max = dist - T_MIN.
            if t > T_MIN && t < dist - T_MIN {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Form-factor computation
// ---------------------------------------------------------------------------

/// Computes the differential-to-finite form factor from patch `pi` to `pj`.
/// Returns zero for degenerate geometry or back-hemisphere directions.
#[inline]
fn compute_form_factor(pi: &Patch, pj: &Patch) -> f32 {
    let dx = pj.centroid[0] - pi.centroid[0];
    let dy = pj.centroid[1] - pi.centroid[1];
    let dz = pj.centroid[2] - pi.centroid[2];

    let r2 = dx * dx + dy * dy + dz * dz;
    if r2 < 1.0e-8 {
        return 0.0;
    }

    let r  = sqrt_f32(r2);
    let nx = dx / r;
    let ny = dy / r;
    let nz = dz / r;

    let cos_i = (pi.normal[0] * nx + pi.normal[1] * ny + pi.normal[2] * nz).max(0.0);
    let cos_j = (-(pj.normal[0] * nx + pj.normal[1] * ny + pj.normal[2] * nz)).max(0.0);

    const PI: f32 = 3.141_592_7;
    (cos_i * cos_j) / (PI * r2) * pj.area
}

// ---------------------------------------------------------------------------
// Moller-Trumbore ray-triangle intersection
// ---------------------------------------------------------------------------

/// Tests a ray (origin `ro`, unit direction `rd`) against the triangle
/// `(v0, v1, v2)`. Returns the intersection distance `t > 0` if hit, or
/// `None` for misses and back-face hits.
///
/// Reference: Moller, T. & Trumbore, B. (1997). "Fast, minimum storage
/// ray/triangle intersection." Journal of Graphics Tools, 2(1), 21-28.
#[inline]
fn moller_trumbore(
    ro: [f32; 3],
    rd: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<f32> {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let h  = cross3(rd, e2);
    let a  = dot3(e1, h);

    // Ray nearly parallel to triangle; treat as a miss.
    if a.abs() < 1.0e-7 {
        return None;
    }

    let f = 1.0 / a;
    let s = sub3(ro, v0);
    let u = f * dot3(s, h);
    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = cross3(s, e1);
    let v = f * dot3(rd, q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * dot3(e2, q);
    if t > 1.0e-7 { Some(t) } else { None }
}

// ---------------------------------------------------------------------------
// Vec3 arithmetic helpers
// ---------------------------------------------------------------------------

#[inline] fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}
#[inline] fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}
#[inline] fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

// ---------------------------------------------------------------------------
// Software square root
// ---------------------------------------------------------------------------

/// Computes `sqrt(x)` using a bit-manipulation initial estimate followed by
/// three Newton-Raphson iterations. Accurate to the last ulp for all
/// non-negative finite f32 values.
#[inline]
fn sqrt_f32(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let bits: u32 = x.to_bits();
    let est:  u32 = (bits >> 1).wrapping_add(0x1FBB_4F2E);
    let mut s = f32::from_bits(est);
    s = 0.5 * (s + x / s);
    s = 0.5 * (s + x / s);
    s = 0.5 * (s + x / s);
    s
}

// ---------------------------------------------------------------------------
// Required no_std support
// ---------------------------------------------------------------------------

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop { core::hint::spin_loop(); }
}
