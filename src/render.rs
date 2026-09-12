//! Software ray caster and PNG output.
//!
//! Renders the radiosity solution using exact Moller-Trumbore ray-triangle
//! intersection against the tessellated patch triangles. Each pixel fires one
//! primary ray; the closest hit determines the pixel colour. No anti-aliasing
//! is performed (one sample per pixel is standard for a POC).
//!
//! Tone mapping uses the extended Reinhard operator. Gamma correction uses the
//! standard sRGB transfer function.

use anyhow::Result;
use image::{ImageBuffer, Rgb};

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

/// Returns the intersection distance `t > 0` for a ray `(ro, rd)` against
/// the triangle `(v0, v1, v2)`, or `None` on a miss.
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
    if a.abs() < 1.0e-7 { return None; }

    let f = 1.0 / a;
    let s = sub3(ro, v0);
    let u = f * dot3(s, h);
    if !(0.0..=1.0).contains(&u) { return None; }

    let q = cross3(s, e1);
    let v = f * dot3(rd, q);
    if v < 0.0 || u + v > 1.0 { return None; }

    let t = f * dot3(e2, q);
    if t > 1.0e-4 { Some(t) } else { None }
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

    let camera = Camera::cornell_default(width, height);

    // White-point from scene luminance maximum (Rec. 709 coefficients).
    let white = radiosities.iter()
        .map(|&[r, g, b]| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let ro = camera.eye;
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    for py in 0..height {
        let rd = camera.ray_dir(0, py); // direction for px=0; overridden per pixel
        let _ = rd; // computed per pixel below

        for px in 0..width {
            let rd = camera.ray_dir(px, py);

            // Pass 1: find the nearest hit patch.
            let mut best_t   = f32::INFINITY;
            let mut best_idx = usize::MAX;

            for (i, patch) in patches.iter().enumerate() {
                // Test both windings: OBJ normals point inward; the reverse
                // winding catches patches whose normal faces away from the eye.
                let t = moller_trumbore(ro, rd, patch.v0, patch.v1, patch.v2)
                    .or_else(|| moller_trumbore(ro, rd, patch.v2, patch.v1, patch.v0));
                if let Some(t) = t && t < best_t {
                    best_t   = t;
                    best_idx = i;
                }
            }

            // Pass 2: inverse-distance-weighted blend of coplanar nearby
            // patches. Eliminates the hard step at patch boundaries without
            // altering patch count or device computation.
            //
            // Radius (0.3 m) covers a 3x3 neighbourhood at 0.1 m patch
            // spacing. The minimum clamp on d^2 prevents division divergence
            // when the hit point coincides with a patch centroid.
            let best_rgb = if best_idx < patches.len() {
                let hit_p = [
                    ro[0] + best_t * rd[0],
                    ro[1] + best_t * rd[1],
                    ro[2] + best_t * rd[2],
                ];
                let hit_n = patches[best_idx].abi.normal;

                // Clamp: (patch_size/4)^2 with patch_size = 0.1 m.
                const EPS_D2:  f32 = 6.25e-4;
                // Cutoff: (3 * patch_size)^2.
                const R_MAX_SQ: f32 = 9.0e-2;

                let mut w_sum = 0.0f32;
                let mut rgb   = [0.0f32; 3];

                for (patch, &rad) in patches.iter().zip(radiosities.iter()) {
                    // Reject patches on different surfaces.
                    if dot3(patch.abi.normal, hit_n) < 0.98 { continue; }
                    let dp = sub3(patch.abi.centroid, hit_p);
                    let d2 = dot3(dp, dp);
                    if d2 > R_MAX_SQ { continue; }
                    let w = 1.0 / (d2 + EPS_D2);
                    w_sum    += w;
                    rgb[0]   += w * rad[0];
                    rgb[1]   += w * rad[1];
                    rgb[2]   += w * rad[2];
                }

                if w_sum > 0.0 {
                    [rgb[0] / w_sum, rgb[1] / w_sum, rgb[2] / w_sum]
                } else {
                    radiosities[best_idx]
                }
            } else {
                [0.0f32; 3]
            };

            let [r, g, b] = to_srgb(best_rgb, white);
            img.put_pixel(px, py, Rgb([r, g, b]));
        }

        if py % 64 == 0 {
            eprintln!("  row {py}/{height}");
        }
    }

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
