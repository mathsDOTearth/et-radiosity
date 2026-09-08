//! Scene loading and patch discretisation.
//!
//! Loads a Wavefront OBJ file (with accompanying MTL), tessellates each face
//! into a regular grid of sub-patches, and assigns reflectance and emission
//! from the material. Returns two collections:
//!
//! - [`ScenePatch`]: tessellated patches for the radiosity solve, each storing
//!   triangle vertices `v0/v1/v2` for exact ray-triangle intersection in the
//!   renderer.
//! - [`OccluderTriangle`]: un-tessellated face triangles from non-wall geometry
//!   (the box and sphere). These are uploaded to the device for inter-patch
//!   visibility testing in the form-factor kernel.

use anyhow::{Context, Result, bail};
use radiosity_abi::Patch;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A host-side scene patch, extending the ABI [`Patch`] with material data
/// and triangle vertices for the software ray caster.
#[derive(Clone, Debug)]
pub struct ScenePatch {
    /// ABI-compatible geometry descriptor uploaded to the device.
    pub abi: Patch,
    /// Diffuse reflectance, RGB in [0, 1].
    pub reflectance: [f32; 3],
    /// Self-emission radiance, RGB (W/m^2/sr). Non-zero only for light sources.
    pub emission: [f32; 3],
    /// Triangle vertices in world space (for Moller-Trumbore in the renderer).
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    /// Material name, retained for diagnostics.
    #[allow(dead_code)]
    pub material: String,
}

/// An un-tessellated face triangle used for shadow-ray visibility testing.
///
/// Only object geometry (non-wall faces) is collected here; the convex room
/// walls never occlude each other in a Cornell box.
#[derive(Clone, Copy, Debug)]
pub struct OccluderTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
}

/// Materials that belong to the room envelope and never act as occluders.
/// Any material name NOT in this set is treated as interior geometry.
const WALL_MATERIALS: &[&str] = &["white", "red", "green", "light"];

fn is_wall_material(name: &str) -> bool {
    WALL_MATERIALS.contains(&name)
}

// ---------------------------------------------------------------------------
// Scene load result
// ---------------------------------------------------------------------------

pub struct Scene {
    pub patches:   Vec<ScenePatch>,
    pub occluders: Vec<OccluderTriangle>,
}

// ---------------------------------------------------------------------------
// OBJ loading
// ---------------------------------------------------------------------------

/// Loads `path` as a Wavefront OBJ file, tessellates all faces into patches
/// no larger than `max_patch_side` metres, and returns the patch list together
/// with the occluder triangle set.
pub fn load_obj(path: &std::path::Path, max_patch_side: f32) -> Result<Scene> {
    let (models, materials_result) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            single_index:  true,
            triangulate:   true,
            ignore_points: true,
            ignore_lines:  true,
        },
    )
    .with_context(|| format!("loading OBJ from {}", path.display()))?;

    let materials = materials_result
        .with_context(|| "loading MTL file referenced by OBJ")?;

    let mut patches:   Vec<ScenePatch>      = Vec::new();
    let mut occluders: Vec<OccluderTriangle> = Vec::new();

    for model in &models {
        let mesh  = &model.mesh;
        let verts = &mesh.positions;

        let mat_id = mesh.material_id.unwrap_or(usize::MAX);
        let (reflectance, emission, mat_name) = if mat_id < materials.len() {
            let m  = &materials[mat_id];
            let kd = m.diffuse.unwrap_or([0.0; 3]);
            // tobj 4.x parses `Ke` into the dedicated `emissive` field.
            let ke = m.emissive.unwrap_or([0.0; 3]);
            (kd, ke, m.name.clone())
        } else {
            ([0.5f32; 3], [0.0f32; 3], String::from("default"))
        };

        let is_wall = is_wall_material(&mat_name);

        for face_idx in 0..mesh.indices.len() / 3 {
            let base = face_idx * 3;
            let i0   = mesh.indices[base]     as usize;
            let i1   = mesh.indices[base + 1] as usize;
            let i2   = mesh.indices[base + 2] as usize;

            let v0 = [verts[i0*3], verts[i0*3+1], verts[i0*3+2]];
            let v1 = [verts[i1*3], verts[i1*3+1], verts[i1*3+2]];
            let v2 = [verts[i2*3], verts[i2*3+1], verts[i2*3+2]];

            let cross      = cross3(sub3(v1, v0), sub3(v2, v0));
            let area_total = len3(cross) * 0.5;
            if area_total < 1.0e-12 { continue; }
            let normal = normalise3(cross);

            // Non-wall faces are also recorded as occluder triangles (original
            // un-tessellated geometry) for the shadow-ray test in the kernel.
            if !is_wall {
                occluders.push(OccluderTriangle { v0, v1, v2 });
            }

            // Tessellate: subdivide along the longest edge.
            let longest = len3(sub3(v1, v0))
                .max(len3(sub3(v2, v0)))
                .max(len3(sub3(v2, v1)));
            let divs = ((longest / max_patch_side).ceil() as usize).max(1);

            tessellate_triangle(
                v0, v1, v2, normal, area_total, divs,
                reflectance, emission, &mat_name,
                &mut patches,
            );
        }
    }

    if patches.is_empty() {
        bail!("OBJ contains no usable faces");
    }

    Ok(Scene { patches, occluders })
}

// ---------------------------------------------------------------------------
// Triangle tessellation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn tessellate_triangle(
    v0: [f32; 3], v1: [f32; 3], v2: [f32; 3],
    normal:      [f32; 3],
    area_total:  f32,
    divs:        usize,
    reflectance: [f32; 3],
    emission:    [f32; 3],
    material:    &str,
    out:         &mut Vec<ScenePatch>,
) {
    let sub_area = area_total / (divs * divs) as f32;

    for row in 0..divs {
        for col in 0..(divs - row) {
            push_sub(v0, v1, v2, normal, sub_area, reflectance, emission, material, out,
                bary_lower(row, col, divs));

            if col + 1 < divs - row {
                push_sub(v0, v1, v2, normal, sub_area, reflectance, emission, material, out,
                    bary_upper(row, col, divs));
            }
        }
    }
}

fn bary_lower(row: usize, col: usize, divs: usize) -> [(f32, f32); 3] {
    let d = divs as f32;
    let (r, c) = (row as f32, col as f32);
    [(c/d, r/d), ((c+1.0)/d, r/d), (c/d, (r+1.0)/d)]
}

fn bary_upper(row: usize, col: usize, divs: usize) -> [(f32, f32); 3] {
    let d = divs as f32;
    let (r, c) = (row as f32, col as f32);
    [((c+1.0)/d, r/d), ((c+1.0)/d, (r+1.0)/d), (c/d, (r+1.0)/d)]
}

fn bary_interp(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], u: f32, v: f32) -> [f32; 3] {
    let w = 1.0 - u - v;
    [ w*v0[0] + u*v1[0] + v*v2[0],
      w*v0[1] + u*v1[1] + v*v2[1],
      w*v0[2] + u*v1[2] + v*v2[2] ]
}

#[allow(clippy::too_many_arguments)]
fn push_sub(
    v0: [f32; 3], v1: [f32; 3], v2: [f32; 3],
    normal:      [f32; 3],
    sub_area:    f32,
    reflectance: [f32; 3],
    emission:    [f32; 3],
    material:    &str,
    out:         &mut Vec<ScenePatch>,
    bary:        [(f32, f32); 3],
) {
    let p0 = bary_interp(v0, v1, v2, bary[0].0, bary[0].1);
    let p1 = bary_interp(v0, v1, v2, bary[1].0, bary[1].1);
    let p2 = bary_interp(v0, v1, v2, bary[2].0, bary[2].1);
    let centroid = [
        (p0[0] + p1[0] + p2[0]) / 3.0,
        (p0[1] + p1[1] + p2[1]) / 3.0,
        (p0[2] + p1[2] + p2[2]) / 3.0,
    ];
    out.push(ScenePatch {
        abi: Patch { centroid, normal, area: sub_area, _pad: 0.0 },
        reflectance,
        emission,
        v0: p0, v1: p1, v2: p2,
        material: material.to_owned(),
    });
}

// ---------------------------------------------------------------------------
// Vec3 helpers
// ---------------------------------------------------------------------------

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
}
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn len3(a: [f32; 3]) -> f32 {
    (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt()
}
fn normalise3(a: [f32; 3]) -> [f32; 3] {
    let l = len3(a);
    if l < 1.0e-12 { return [0.0; 3]; }
    [a[0]/l, a[1]/l, a[2]/l]
}
