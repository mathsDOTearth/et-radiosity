//! Software ray caster and PNG output.
//!
//! Renders the radiosity solution using exact Moller-Trumbore ray-triangle
//! intersection against the tessellated patch triangles. Each pixel fires one
//! primary ray; the closest hit determines the pixel colour. No anti-aliasing
//! is performed (one sample per pixel is standard for a POC).
//!
//! Gouraud shading: a per-vertex radiosity is precomputed by averaging the
//! radiosity of all patches incident on each unique vertex. The Moller-Trumbore
//! barycentric coordinates are used to interpolate those vertex values across
//! each hit triangle, eliminating the hard step artefact at patch boundaries
//! without altering the radiosity solve or device compute time.
//!
//! Tone mapping uses the extended Reinhard operator. Gamma correction uses the
//! standard sRGB transfer function.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use image::{ImageBuffer, Rgb};
use rayon::prelude::*;

use crate::scene::ScenePatch;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

struct Camera {
    eye:          [f32; 3],
    forward:      [f32; 3],
    right:        [f32; 3],
    up:           [f32; 3],
    tan_half_fov: f32,
    width:        u32,
    height:       u32,
}

impl Camera {
    fn cornell_default(width: u32, height: u32) -> Self {
        Camera {
            eye:          [0.5, 0.5, -1.4],
            forward:      [0.0, 0.0,  1.0],
            right:        [1.0, 0.0,  0.0],
            up:           [0.0, 1.0,  0.0],
            tan_half_fov: (22.5f32).to_radians().tan(), // 45-degree FOV
            width,
            height,
        }
    }

    fn ray_dir(&self, px: u32, py: u32) -> [f32; 3] {
        let aspect = self.width as f32 / self.height as f32;
        let ndc_x  =  (px as f32 + 0.5) / self.width  as f32 * 2.0 - 1.0;
        let ndc_y  = -(py as f32 + 0.5) / self.height as f32 * 2.0 + 1.0;
        let dx     = ndc_x * aspect * self.tan_half_fov;
        let dy     = ndc_y          * self.tan_half_fov;
        normalise3([
            self.forward[0] + dx * self.right[0] + dy * self.up[0],
            self.forward[1] + dx * self.right[1] + dy * self.up[1],
            self.forward[2] + dx * self.right[2] + dy * self.up[2],
        ])
    }
}

// ---------------------------------------------------------------------------
// Moller-Trumbore ray-triangle intersection
// ---------------------------------------------------------------------------

/// Returns `(t, u, v)` for a ray `(ro, rd)` against the triangle `(v0, v1,
/// v2)`, or `None` on a miss.  The barycentric weight for `v0` is `1-u-v`,
/// for `v1` is `u`, and for `v2` is `v`.
fn moller_trumbore(
    ro: [f32; 3],
    rd: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> Option<(f32, f32, f32)> {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let h  = cross3(rd, e2);
    let a  = dot3(e1, h);
    if a.abs() < 1.0e-7 { return None; }

    let f = 1.0 / a;
    let s = sub3(ro, v0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) { return None; }

    let q = cross3(s, e1);
    let v = f * dot3(rd, q);
    if v < 0.0 || u + v > 1.0 { return None; }

    let t = f * dot3(e2, q);
    if t > 1.0e-4 { Some((t, u, v)) } else { None }
}

// ---------------------------------------------------------------------------
// Per-vertex radiosity (Gouraud shading precomputation)
// ---------------------------------------------------------------------------

/// Vertex index triple per patch: `[vi0, vi1, vi2]` for `patch.v0/v1/v2`.
pub type PatchVerts = [usize; 3];

/// Deduplicates triangle vertices by position (1 mm quantisation) and
/// computes a per-vertex radiosity as the average of all incident patches.
///
/// Returns `(vertex_radiosity, patch_vert_indices)`.  `vertex_radiosity[vi]`
/// is the radiosity attributed to the unique vertex at index `vi`; indexing
/// it from the barycentric coordinates returned by `moller_trumbore` gives
/// smooth Gouraud-shaded colour at every rendered pixel.
pub fn build_vertex_radiosity(
    patches:     &[ScenePatch],
    radiosities: &[[f32; 3]],
) -> (Vec<[f32; 3]>, Vec<PatchVerts>) {
    // 1 mm quantisation: vertices within 1 mm of each other merge.
    const GRID: f32 = 1.0e-3;

    let mut key_to_vi: HashMap<(i32, i32, i32), usize> = HashMap::new();
    let mut vert_acc:  Vec<[f64; 3]> = Vec::new();  // f64 accumulator avoids
    let mut vert_cnt:  Vec<u32>      = Vec::new();  // cancellation for bright scenes
    let mut patch_vi:  Vec<PatchVerts> = Vec::with_capacity(patches.len());

    let quantize = |x: f32| -> i32 { (x / GRID).round() as i32 };

    for (pi, patch) in patches.iter().enumerate() {
        let rad = radiosities[pi];
        let mut idx = [0usize; 3];

        for (j, &v) in [patch.v0, patch.v1, patch.v2].iter().enumerate() {
            let key = (quantize(v[0]), quantize(v[1]), quantize(v[2]));
            let vi  = *key_to_vi.entry(key).or_insert_with(|| {
                let i = vert_acc.len();
                vert_acc.push([0.0; 3]);
                vert_cnt.push(0);
                i
            });
            vert_acc[vi][0] += rad[0] as f64;
            vert_acc[vi][1] += rad[1] as f64;
            vert_acc[vi][2] += rad[2] as f64;
            vert_cnt[vi]    += 1;
            idx[j] = vi;
        }

        patch_vi.push(idx);
    }

    let vert_rad: Vec<[f32; 3]> = vert_acc.iter().zip(vert_cnt.iter())
        .map(|(sum, &c)| {
            if c > 0 {
                let f = 1.0 / c as f64;
                [(sum[0]*f) as f32, (sum[1]*f) as f32, (sum[2]*f) as f32]
            } else {
                [0.0; 3]
            }
        })
        .collect();

    (vert_rad, patch_vi)
}

// ---------------------------------------------------------------------------
// Tone mapping and gamma
// ---------------------------------------------------------------------------

/// Reinhard extended tone mapping (Reinhard et al., 2002).
#[inline]
fn reinhard(x: f32, white: f32) -> f32 {
    x * (1.0 + x / (white * white)) / (1.0 + x)
}

/// Maps a linear HDR RGB triplet to a gamma-corrected sRGB byte triplet.
fn to_srgb(rgb: [f32; 3], white: f32) -> [u8; 3] {
    let gamma = |x: f32| x.powf(1.0 / 2.2).clamp(0.0, 1.0);
    [
        (gamma(reinhard(rgb[0].max(0.0), white)) * 255.0) as u8,
        (gamma(reinhard(rgb[1].max(0.0), white)) * 255.0) as u8,
        (gamma(reinhard(rgb[2].max(0.0), white)) * 255.0) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Render entry point
// ---------------------------------------------------------------------------

/// Renders the radiosity solution to a PNG file.
pub fn render_to_png(
    patches:     &[ScenePatch],
    radiosities: &[[f32; 3]],
    width:       u32,
    height:      u32,
    output_path: &std::path::Path,
) -> Result<()> {
    assert_eq!(patches.len(), radiosities.len());
    eprintln!("Render: {width}x{height}, {} patches...", patches.len());

    // Precompute Gouraud vertex radiosity.
    let (vert_rad, patch_vi) = build_vertex_radiosity(patches, radiosities);
    eprintln!("  {} unique vertices (Gouraud shading)", vert_rad.len());

    let camera = Camera::cornell_default(width, height);

    // White-point from scene luminance maximum (Rec. 709 coefficients).
    let white = radiosities.iter()
        .map(|&[r, g, b]| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let ro = camera.eye;

    // Pre-allocate the pixel buffer: width * height * 3 bytes (RGB).
    // Rows are computed in parallel; each row owns a disjoint slice of the
    // buffer so no locking is required.
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    let rows_done  = AtomicU32::new(0);

    pixels
        .par_chunks_mut((width * 3) as usize)
        .enumerate()
        .for_each(|(py, row_buf)| {
            for px in 0..width {
                let rd = camera.ray_dir(px, py as u32);

                // Find the nearest hit, recording hit-triangle index and the
                // Moller-Trumbore barycentric coordinates (u, v).
                let mut best_t    = f32::INFINITY;
                let mut best_idx  = usize::MAX;
                let mut best_u    = 0.0f32;
                let mut best_v    = 0.0f32;
                // Indicates the reverse-winding test produced the best hit.
                let mut best_flip = false;

                for (i, patch) in patches.iter().enumerate() {
                    // Test the triangle's natural winding first.
                    if let Some((t, u, v)) = moller_trumbore(ro, rd, patch.v0, patch.v1, patch.v2) {
                        if t < best_t {
                            best_t    = t;
                            best_idx  = i;
                            best_u    = u;
                            best_v    = v;
                            best_flip = false;
                        }
                    }
                    // Test the reverse winding: the OBJ normals point inward, so
                    // patches whose normal faces away from the eye are hit only via
                    // this path.
                    if let Some((t, u, v)) = moller_trumbore(ro, rd, patch.v2, patch.v1, patch.v0) {
                        if t < best_t {
                            best_t    = t;
                            best_idx  = i;
                            best_u    = u;
                            best_v    = v;
                            best_flip = true;
                        }
                    }
                }

                let best_rgb = if best_idx < patches.len() {
                    let [vi0, vi1, vi2] = patch_vi[best_idx];

                    // Barycentric weights for the three vertices.
                    //
                    // Natural winding (v0, v1, v2): u -> v1, v -> v2, 1-u-v -> v0.
                    // Flipped winding  (v2, v1, v0): MT's "v1" is original v1
                    // (unchanged), "v2" is original v0, "v0" is original v2.
                    // So: u -> vi1, v -> vi0, 1-u-v -> vi2.
                    let (w0, w1, w2) = if best_flip {
                        (best_v, best_u, 1.0 - best_u - best_v)
                    } else {
                        (1.0 - best_u - best_v, best_u, best_v)
                    };

                    let r0 = vert_rad[vi0];
                    let r1 = vert_rad[vi1];
                    let r2 = vert_rad[vi2];
                    [
                        w0 * r0[0] + w1 * r1[0] + w2 * r2[0],
                        w0 * r0[1] + w1 * r1[1] + w2 * r2[1],
                        w0 * r0[2] + w1 * r1[2] + w2 * r2[2],
                    ]
                } else {
                    [0.0f32; 3]
                };

                let [r, g, b] = to_srgb(best_rgb, white);
                let base = (px * 3) as usize;
                row_buf[base]     = r;
                row_buf[base + 1] = g;
                row_buf[base + 2] = b;
            }

            // Progress reporting: print every 64 completed rows.
            let done = rows_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 64 == 0 || done == height {
                eprintln!("  {done}/{height} rows done");
            }
        });

    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, pixels)
            .expect("pixel buffer dimensions match image size");

    img.save(output_path)
        .map_err(|e| anyhow::anyhow!("saving PNG: {e}"))?;
    eprintln!("  saved to {}", output_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Vec3 helpers
// ---------------------------------------------------------------------------

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn normalise3(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    if l < 1.0e-12 { return a; }
    [a[0]/l, a[1]/l, a[2]/l]
}
