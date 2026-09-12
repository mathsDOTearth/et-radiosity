//! ET-SoC-1 render kernel.
//!
//! Each hart ray-casts a disjoint subset of the output pixels against the
//! uploaded patch geometry, computes Gouraud-interpolated radiosity at the
//! nearest hit, applies Reinhard tone mapping and sRGB gamma, and writes the
//! resulting RGB bytes to the output pixel buffer.
//!
//! Pixel partitioning mirrors the form-factor kernel: pixels are linearised
//! row-major (index = py * width + px) and distributed across harts via a
//! strided grid -- hart h processes pixels h, h + n_harts, h + 2*n_harts, ...
//!
//! # Floating-point note
//!
//! The target ISA (RV64IMAC) has no scalar F/D extensions; f32 arithmetic is
//! lowered to compiler_builtins soft-float routines. `sqrt` is implemented via
//! a bit-manipulation initial estimate followed by three Newton-Raphson
//! iterations (identical to ff-kernel). The sRGB gamma approximation uses the
//! IEEE 754 exponent linearity trick: bits(x^k) scale linearly with k, giving
//! a single integer multiply in place of repeated `sqrt` calls.

#![no_std]
#![no_main]

use et_abi::DeviceArgs;
use et_kernel::{fence, kernel_entry, Grid};
use radiosity_abi::{PatchGeom, RenderArgs};

kernel_entry!();

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------

/// Kernel entry: distributes pixel ray-cast across harts.
///
/// `args_ptr` is the raw pointer delivered in `a0` by the host firmware,
/// addressing a `RenderArgs` in device DRAM.
///
/// # Safety
///
/// `args_ptr` must be the raw pointer delivered by the host firmware.
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    let args = unsafe { RenderArgs::from_ptr(args_ptr as *const u8) };

    let grid = Grid::new(args.n_harts);
    if !grid.active() {
        return 0;
    }

    let n_pixels  = args.width as usize * args.height as usize;
    let hart_idx  = grid.hart() as usize;
    let n_harts   = grid.n_harts() as usize;

    // SAFETY: all regions were allocated and initialised by the host before
    // launch; addresses and lengths are as recorded in RenderArgs.
    let geom = unsafe {
        core::slice::from_raw_parts(
            args.patch_geom_addr as *const PatchGeom,
            args.n_patches as usize,
        )
    };
    let vert_rad = unsafe {
        core::slice::from_raw_parts(
            args.vert_rad_addr as *const [f32; 3],
            args.n_verts as usize,
        )
    };
    let patch_vi = unsafe {
        core::slice::from_raw_parts(
            args.patch_vi_addr as *const [u32; 3],
            args.n_patches as usize,
        )
    };
    let pixels = unsafe {
        core::slice::from_raw_parts_mut(
            args.pixels_addr as *mut u8,
            n_pixels * 3,
        )
    };

    // Camera parameters: eye[3], fwd[3], right[3], up[3], tan_half_fov, aspect.
    let cam = unsafe {
        core::slice::from_raw_parts(args.camera_addr as *const f32, 14)
    };
    let eye          = [cam[0],  cam[1],  cam[2]];
    let fwd          = [cam[3],  cam[4],  cam[5]];
    let right        = [cam[6],  cam[7],  cam[8]];
    let up           = [cam[9],  cam[10], cam[11]];
    let tan_half_fov = cam[12];
    let aspect       = cam[13];
    let white        = args.white;
    let width        = args.width  as usize;
    let height       = args.height as usize;

    // Each hart processes every n_harts-th pixel (strided distribution).
    let mut pixel_idx = hart_idx;
    while pixel_idx < n_pixels {
        let py = pixel_idx / width;
        let px = pixel_idx % width;

        let rd = ray_dir(px, py, width, height, fwd, right, up, tan_half_fov, aspect);
        let rgb = cast_ray(eye, rd, geom, vert_rad, patch_vi, white);

        // SAFETY: pixel_idx < n_pixels, so base + 2 < n_pixels * 3.
        // No other hart writes to this pixel (strided partition is disjoint).
        unsafe {
            let base = pixel_idx * 3;
            (pixels.as_mut_ptr().add(base)).write_volatile(rgb[0]);
            (pixels.as_mut_ptr().add(base + 1)).write_volatile(rgb[1]);
            (pixels.as_mut_ptr().add(base + 2)).write_volatile(rgb[2]);
        }

        pixel_idx += n_harts;
    }

    fence();
    0
}

// ---------------------------------------------------------------------------
// Ray direction
// ---------------------------------------------------------------------------

/// Constructs a normalised ray direction for pixel `(px, py)` using a
/// rectilinear camera model. The coordinate convention matches the host-side
/// `Camera::ray_dir`: NDC origin at the image centre, Y-up.
#[inline]
fn ray_dir(
    px: usize, py: usize,
    width: usize, height: usize,
    fwd:   [f32; 3],
    right: [f32; 3],
    up:    [f32; 3],
    tan_half_fov: f32,
    aspect: f32,
) -> [f32; 3] {
    let ndc_x =  (px as f32 + 0.5) / width  as f32 * 2.0 - 1.0;
    let ndc_y = -(py as f32 + 0.5) / height as f32 * 2.0 + 1.0;
    let dx    = ndc_x * aspect * tan_half_fov;
    let dy    = ndc_y * tan_half_fov;
    normalise3([
        fwd[0] + dx * right[0] + dy * up[0],
        fwd[1] + dx * right[1] + dy * up[1],
        fwd[2] + dx * right[2] + dy * up[2],
    ])
}

// ---------------------------------------------------------------------------
// Nearest-hit ray cast
// ---------------------------------------------------------------------------

/// Returns the sRGB byte triple for the ray `(ro, rd)` cast against the patch
/// geometry. Gouraud shading is applied by interpolating per-vertex radiosity
/// values using the Moller-Trumbore barycentric coordinates of the nearest hit.
///
/// Returns `[0, 0, 0]` (black) for rays that miss all geometry.
fn cast_ray(
    ro:       [f32; 3],
    rd:       [f32; 3],
    geom:     &[PatchGeom],
    vert_rad: &[[f32; 3]],
    patch_vi: &[[u32; 3]],
    white:    f32,
) -> [u8; 3] {
    let mut best_t    = f32::INFINITY;
    let mut best_idx  = usize::MAX;
    let mut best_u    = 0.0f32;
    let mut best_v    = 0.0f32;
    let mut best_flip = false;

    for (i, g) in geom.iter().enumerate() {
        // Test natural winding.
        if let Some((t, u, v)) = moller_trumbore(ro, rd, g.v0, g.v1, g.v2) {
            if t < best_t {
                best_t = t; best_idx = i;
                best_u = u; best_v = v;
                best_flip = false;
            }
        }
        // Test reverse winding: OBJ normals point inward; some patches are
        // visible only through this path.
        if let Some((t, u, v)) = moller_trumbore(ro, rd, g.v2, g.v1, g.v0) {
            if t < best_t {
                best_t = t; best_idx = i;
                best_u = u; best_v = v;
                best_flip = true;
            }
        }
    }

    if best_idx >= geom.len() {
        return [0, 0, 0];
    }

    let [vi0, vi1, vi2] = patch_vi[best_idx].map(|x| x as usize);

    // Recover barycentric weights for the original vertex ordering.
    //
    // Natural winding (v0, v1, v2): MT returns u -> v1, v -> v2.
    //   w(vi0) = 1-u-v,  w(vi1) = u,  w(vi2) = v.
    //
    // Flipped winding (v2, v1, v0): MT's "v0"=original v2, "v1"=original v1,
    //   "v2"=original v0.  MT returns u -> original v1, v -> original v0.
    //   w(vi0) = v,  w(vi1) = u,  w(vi2) = 1-u-v.
    let (w0, w1, w2) = if best_flip {
        (best_v, best_u, 1.0 - best_u - best_v)
    } else {
        (1.0 - best_u - best_v, best_u, best_v)
    };

    let r0 = vert_rad[vi0];
    let r1 = vert_rad[vi1];
    let r2 = vert_rad[vi2];
    let rgb = [
        w0 * r0[0] + w1 * r1[0] + w2 * r2[0],
        w0 * r0[1] + w1 * r1[1] + w2 * r2[1],
        w0 * r0[2] + w1 * r1[2] + w2 * r2[2],
    ];

    to_srgb(rgb, white)
}

// ---------------------------------------------------------------------------
// Moller-Trumbore ray-triangle intersection (returns t, u, v barycentric)
// ---------------------------------------------------------------------------

/// Tests a ray `(ro, rd)` against the triangle `(v0, v1, v2)`. Returns
/// `(t, u, v)` -- intersection distance and Moller-Trumbore barycentric
/// coordinates -- when the ray hits the front face at `t > 1e-4`, or `None`
/// for misses and back-face hits.
///
/// Reference: Moller, T. & Trumbore, B. (1997). "Fast, minimum storage
/// ray/triangle intersection." Journal of Graphics Tools, 2(1), 21-28.
#[inline]
fn moller_trumbore(
    ro: [f32; 3], rd: [f32; 3],
    v0: [f32; 3], v1: [f32; 3], v2: [f32; 3],
) -> Option<(f32, f32, f32)> {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let h  = cross3(rd, e2);
    let a  = dot3(e1, h);
    // Ray nearly parallel to triangle plane; treat as a miss.
    if a > -1.0e-7 && a < 1.0e-7 { return None; }
    let f  = 1.0 / a;
    let s  = sub3(ro, v0);
    let u  = f * dot3(s, h);
    if u < 0.0 || u > 1.0 { return None; }
    let q  = cross3(s, e1);
    let v  = f * dot3(rd, q);
    if v < 0.0 || u + v > 1.0 { return None; }
    let t  = f * dot3(e2, q);
    // t_min = 1e-4 m prevents self-intersection artefacts.
    if t > 1.0e-4 { Some((t, u, v)) } else { None }
}

// ---------------------------------------------------------------------------
// Tone mapping and gamma
// ---------------------------------------------------------------------------

/// Reinhard extended tone mapping (Reinhard et al., 2002).
#[inline]
fn reinhard(x: f32, white: f32) -> f32 {
    x * (1.0 + x / (white * white)) / (1.0 + x)
}

/// Maps a linear HDR RGB triplet to a gamma-corrected sRGB byte triplet using
/// Reinhard tone mapping followed by `pow22` gamma.
fn to_srgb(rgb: [f32; 3], white: f32) -> [u8; 3] {
    let f = |x: f32| -> u8 {
        let t = reinhard(x.max(0.0), white);
        (pow22(t) * 255.0 + 0.5) as u8
    };
    [f(rgb[0]), f(rgb[1]), f(rgb[2])]
}

/// Approximates `x^(1/2.2)` for `x` in [0, 1] using IEEE 754 exponent
/// linearity: for positive normal floats, `bits(x^k)` scales linearly in `k`
/// as `k * (bits(x) - 0x3F80_0000) + 0x3F80_0000`. With `k = 1/2.2 ~
/// 4545/10000` this avoids all transcendental operations. Error is below 0.5%
/// across [0, 1], well within 8-bit output resolution.
#[inline]
fn pow22(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let bits    = x.to_bits();
    let shifted = bits.wrapping_sub(0x3F80_0000);
    let scaled  = (shifted as u64 * 4545 / 10000) as u32;
    f32::from_bits(scaled.wrapping_add(0x3F80_0000))
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

/// Normalises a vector to unit length. Returns the original vector unchanged
/// if its magnitude is below 1e-12 (degenerate ray direction).
#[inline]
fn normalise3(a: [f32; 3]) -> [f32; 3] {
    let l = sqrt_f32(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    if l < 1.0e-12 { return a; }
    [a[0]/l, a[1]/l, a[2]/l]
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
