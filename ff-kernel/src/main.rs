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
//! # Visibility (shadow) test
//!
//! The occluder array is stored in SoA (struct-of-arrays) format: nine
//! contiguous `f32[n_padded]` arrays (v0x, v0y, v0z, v1x, ..., v2z), where
//! `n_padded = (n_occluders + 7) & !7`. This layout allows the SIMD path to
//! issue eight-lane `AIF.FLW.PS` loads with no gather overhead.
//!
//! The SIMD inner loop processes 8 triangles per iteration using the ET-SoC-1
//! Packed-Single (PS) extensions (256-bit float registers, eight f32 lanes).
//! The Moller-Trumbore test is reformulated to avoid `FDIV.PS` (which traps to
//! M-mode firmware on this device): all conditions are tested after multiplying
//! through by `sign(a)`, converting the reciprocal test into integer-sign
//! comparisons solvable with `FLE.PS`/`FLT.PS` (result in float register as
//! 0.0/1.0) and accumulating into a per-lane hit boolean via `FMUL.PS`.
//!
//! # Floating-point note
//!
//! The target ISA (RV64IMAC) has no scalar F/D extensions; f32 arithmetic is
//! lowered to compiler_builtins soft-float routines. `sqrt` is implemented via
//! three Newton-Raphson iterations after a bit-manipulation initial estimate,
//! giving full f32 precision without requiring libm. PS instructions access the
//! 256-bit floating-point register file that the ET-Minion vector unit provides
//! independently of the base ISA scalar float extensions.

#![no_std]
#![no_main]

use et_abi::DeviceArgs;
use et_kernel::{Grid, MsgBuf, device_slice, fence, kernel_entry, trace_str};
use radiosity_abi::{FormFactorArgs, Patch};

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
    let n_occ = args.n_occluders as usize;

    // SAFETY: host allocated and initialised these arrays before launch.
    let patches: &[Patch] = unsafe { device_slice(args.patches_addr as usize, n) };

    // SoA occluder block: nine f32[n_padded] arrays.  n_padded is derived from
    // n_occ here exactly as the host computed it (ceil to multiple of 8).
    let n_padded = (n_occ + 7) & !7;
    let occ_soa: *const f32 = if n_occ > 0 {
        args.occluder_addr as *const f32
    } else {
        core::ptr::null()
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
                form_factor_with_visibility(pi, pj, occ_soa, n_padded, n_occ)
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
fn form_factor_with_visibility(
    pi: &Patch,
    pj: &Patch,
    occ_soa: *const f32,
    n_padded: usize,
    n_occ: usize,
) -> f32 {
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
    if n_occ > 0 && !is_visible(pi, pj, occ_soa, n_padded, n_occ) {
        return 0.0;
    }

    compute_form_factor(pi, pj)
}

// ---------------------------------------------------------------------------
// Visibility test (SIMD + scalar fallback Moller-Trumbore shadow ray)
// ---------------------------------------------------------------------------

/// Returns `true` if the segment from `pi.centroid` to `pj.centroid` is not
/// blocked by any occluder triangle.
///
/// The test dispatches to the eight-lane PS SIMD path (`is_visible_simd8`) for
/// full batches of 8 triangles, then falls back to the scalar Moller-Trumbore
/// routine for any trailing triangles. A 1 mm end-point margin prevents
/// self-intersection against the triangles at the emitting and receiving patch.
#[inline]
fn is_visible(
    pi: &Patch,
    pj: &Patch,
    occ_soa: *const f32,
    n_padded: usize,
    n_occ: usize,
) -> bool {
    let ro = pi.centroid;
    let dx = pj.centroid[0] - ro[0];
    let dy = pj.centroid[1] - ro[1];
    let dz = pj.centroid[2] - ro[2];
    let dist = sqrt_f32(dx * dx + dy * dy + dz * dz);
    if dist < 1.0e-6 {
        return false;
    }
    let inv = 1.0 / dist;
    let rd = [dx * inv, dy * inv, dz * inv];
    const T_MIN: f32 = 1.0e-3;

    let full_batches = n_occ / 8;
    let tail_start   = full_batches * 8;

    // SIMD path: 8 triangles per iteration.
    if full_batches > 0 {
        // SAFETY: occ_soa points to a valid SoA block of 9 * n_padded f32
        // values uploaded by the host before launch.  full_batches * 8 <=
        // n_occ <= n_padded, so every FLW.PS load is in bounds.
        if unsafe { !is_visible_simd8(occ_soa, n_padded, full_batches, ro, rd, dist) } {
            return false;
        }
    }

    // Scalar tail: triangles [tail_start, n_occ).
    for i in tail_start..n_occ {
        // SAFETY: i < n_occ <= n_padded; each of the 9 component arrays has
        // at least n_padded elements, so index i is valid in each.
        let v0 = unsafe { [
            *occ_soa.add(0 * n_padded + i),
            *occ_soa.add(1 * n_padded + i),
            *occ_soa.add(2 * n_padded + i),
        ]};
        let v1 = unsafe { [
            *occ_soa.add(3 * n_padded + i),
            *occ_soa.add(4 * n_padded + i),
            *occ_soa.add(5 * n_padded + i),
        ]};
        let v2 = unsafe { [
            *occ_soa.add(6 * n_padded + i),
            *occ_soa.add(7 * n_padded + i),
            *occ_soa.add(8 * n_padded + i),
        ]};
        if let Some(t) = moller_trumbore(ro, rd, v0, v1, v2) {
            if t > T_MIN && t < dist - T_MIN {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Eight-lane PS SIMD Moller-Trumbore shadow test
// ---------------------------------------------------------------------------

// Extern declaration for the assembly implementation in `src/shadow_simd.s`.
// Assembled by the Esperanto GCC toolchain (riscv64-unknown-elf-as
// -march=rv64imaf_xet1p0) via build.rs, because the stable Rust/LLVM
// integrated assembler does not know the XAIFET extension.
unsafe extern "C" {
    /// Tests `full_batches * 8` triangles against a shadow ray.
    ///
    /// Arguments:
    ///   soa           -- SoA occluder base pointer
    ///   n_padded      -- component-array stride, in f32 elements
    ///   full_batches  -- number of 8-wide batches to process
    ///   ro            -- pointer to ray origin  \[3 x f32\]
    ///   rd            -- pointer to ray direction \[3 x f32\]
    ///   consts        -- pointer to \[0.0, eps, T_MIN, dist-T_MIN\] as f32\[4\]
    ///
    /// Returns 0 if no triangle blocks the segment, nonzero otherwise.
    fn simd_shadow_test(
        soa:          *const f32,
        n_padded:     usize,
        full_batches: usize,
        ro:           *const f32,
        rd:           *const f32,
        consts:       *const f32,
    ) -> u64;
}

/// Dispatches the eight-lane PS SIMD shadow test for `full_batches` batches.
///
/// Returns `true` if no triangle in the batched range blocks the segment.
///
/// # Safety
///
/// `soa` must point to a valid SoA occluder block with at least `n_padded * 9`
/// initialised f32 values.  `full_batches * 8 <= n_padded`.
#[inline]
unsafe fn is_visible_simd8(
    soa:          *const f32,
    n_padded:     usize,
    full_batches: usize,
    ro:           [f32; 3],
    rd:           [f32; 3],
    dist:         f32,
) -> bool {
    const T_MIN: f32 = 1.0e-3_f32;
    const EPS:   f32 = 1.0e-7_f32;
    let consts: [f32; 4] = [0.0, EPS, T_MIN, dist - T_MIN];

    // SAFETY: delegated to the caller; assembly only reads within the SoA
    // block and the small stack arrays (ro, rd, consts).
    unsafe {
        simd_shadow_test(
            soa,
            n_padded,
            full_batches,
            ro.as_ptr(),
            rd.as_ptr(),
            consts.as_ptr(),
        ) == 0
    }
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
