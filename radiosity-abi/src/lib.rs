//! Shared ABI types for the form-factor kernel.
//!
//! Both the host (`et-radiosity`) and the device kernel (`ff-kernel`) depend on
//! this crate, so it must compile in a `no_std` environment. All types are
//! `#[repr(C)]` and contain only POD fields, satisfying the requirements of the
//! [`et_abi::DeviceArgs`] safety contract.

#![no_std]

use et_abi::{DeviceArgs, DevicePod};

// ---------------------------------------------------------------------------
// Patch descriptor
// ---------------------------------------------------------------------------

/// A discretised surface patch: the fundamental unit of the radiosity solve.
///
/// Stored in device DRAM and accessed by both the host (for DMA upload) and
/// the form-factor kernel (for form-factor evaluation).
/// Layout: 32 bytes, naturally aligned.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Patch {
    /// World-space centroid, in metres.
    pub centroid: [f32; 3],
    /// Outward-facing unit normal.
    pub normal: [f32; 3],
    /// Surface area, in square metres.
    pub area: f32,
    /// Explicit padding to 32 bytes; must be zero.
    pub _pad: f32,
}

// ---------------------------------------------------------------------------
// Occluder triangle
// ---------------------------------------------------------------------------

/// A triangle used for inter-patch visibility (shadow) testing in the kernel.
///
/// The kernel tests whether the segment between two patch centroids is blocked
/// by any occluder triangle using the Moller-Trumbore algorithm. Only object
/// geometry (the box and sphere) is uploaded as occluders; the convex room
/// walls never occlude each other and are excluded.
///
/// Layout: 48 bytes, each vertex group padded to 16 bytes.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct OccluderTri {
    pub v0: [f32; 3],
    pub _pad0: f32,
    pub v1: [f32; 3],
    pub _pad1: f32,
    pub v2: [f32; 3],
    pub _pad2: f32,
}

// ---------------------------------------------------------------------------
// Kernel launch arguments
// ---------------------------------------------------------------------------

/// Launch arguments for the form-factor kernel.
///
/// The kernel computes the N x N form-factor matrix `F` where entry (i, j) is
/// the fraction of diffuse energy leaving patch i that arrives at patch j,
/// accounting for cosine weighting and inter-patch visibility.
///
/// Layout (all fields at natural alignment, no implicit padding):
///
/// ```text
/// offset  0: patches_addr    -- device address of Patch[n_patches]
/// offset  8: ff_matrix_addr  -- device address of f32[n_patches * n_patches]
/// offset 16: occluder_addr   -- device address of SoA occluder block
/// offset 24: n_patches
/// offset 28: n_harts
/// offset 32: n_occluders
/// offset 36: _pad
/// ```
///
/// The SoA occluder block at `occluder_addr` contains nine contiguous
/// `f32[n_padded]` arrays (where `n_padded = (n_occluders + 7) & !7`):
/// v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z -- in that order.
/// Trailing elements beyond `n_occluders` are zero-padded.
/// This layout allows the PS SIMD kernel to issue eight-lane `FLW.PS` loads
/// without gather instructions.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FormFactorArgs {
    /// Device address of the input patch array (`Patch[n_patches]`).
    pub patches_addr: u64,
    /// Device address of the output form-factor matrix (`f32[n_patches * n_patches]`),
    /// stored row-major: element `(i, j)` is at byte offset `(i * n_patches + j) * 4`.
    pub ff_matrix_addr: u64,
    /// Device address of the SoA occluder block (nine `f32[n_padded]` arrays).
    /// Zero when `n_occluders == 0` (convex enclosure, no visibility testing).
    pub occluder_addr: u64,
    /// Number of patches.
    pub n_patches: u32,
    /// Hart count passed to `Grid::new()`.
    pub n_harts: u32,
    /// Number of occluder triangles. Zero disables visibility testing.
    pub n_occluders: u32,
    /// Explicit padding to maintain 8-byte alignment of subsequent fields.
    pub _pad: u32,
}

// SAFETY: `FormFactorArgs` is `#[repr(C)]`, all fields are unsigned integers,
// the struct contains no padding, and every possible bit pattern is a valid
// (if semantically meaningless) value -- satisfying the `DeviceArgs` contract.
unsafe impl DeviceArgs for FormFactorArgs {}

// SAFETY: `Patch` is `#[repr(C)]`, contains only `f32` fields (no pointers,
// no padding beyond the explicit `_pad` field), and is valid for any bit pattern.
unsafe impl DevicePod for Patch {}

// SAFETY: `OccluderTri` is `#[repr(C)]`, contains only `f32` fields (no pointers,
// no padding beyond the explicit `_pad{0,1,2}` fields), and is valid for any bit pattern.
unsafe impl DevicePod for OccluderTri {}

// ---------------------------------------------------------------------------
// Patch vertex geometry (render kernel)
// ---------------------------------------------------------------------------

/// Vertex positions of a patch triangle, for upload to the render kernel.
///
/// Separate from [`Patch`] so the form-factor kernel ABI is unchanged.
/// Layout: 48 bytes; each vertex group padded to 16 bytes, matching
/// [`OccluderTri`] and maintaining 16-byte alignment within device DRAM.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct PatchGeom {
    pub v0:  [f32; 3],
    pub _p0: f32,
    pub v1:  [f32; 3],
    pub _p1: f32,
    pub v2:  [f32; 3],
    pub _p2: f32,
}

// ---------------------------------------------------------------------------
// Render kernel launch arguments
// ---------------------------------------------------------------------------

/// Arguments passed to the render kernel.
///
/// The host uploads patch vertex geometry, per-vertex radiosity, and per-patch
/// vertex index triples before launching. The kernel writes a packed RGB byte
/// buffer (width * height * 3 bytes, row-major) to `pixels_addr`.
///
/// Layout (all fields at natural alignment, no implicit padding):
///
/// ```text
/// offset  0: patch_geom_addr  -- device address of PatchGeom[n_patches]
/// offset  8: vert_rad_addr    -- device address of [f32; 3][n_verts]
/// offset 16: patch_vi_addr    -- device address of [u32; 3][n_patches]
/// offset 24: pixels_addr      -- device address of u8[width * height * 3]
/// offset 32: camera_addr      -- device address of f32[14]
/// offset 40: white
/// offset 44: width
/// offset 48: height
/// offset 52: n_patches
/// offset 56: n_verts
/// offset 60: n_harts
/// offset 64: _pad
/// ```
///
/// Camera buffer at `camera_addr` holds 14 consecutive f32 values:
/// eye[3], fwd[3], right[3], up[3], tan_half_fov, aspect.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct RenderArgs {
    /// Device address of the vertex geometry array (`PatchGeom[n_patches]`).
    pub patch_geom_addr: u64,
    /// Device address of per-vertex radiosity (`[f32; 3][n_verts]`).
    pub vert_rad_addr:   u64,
    /// Device address of patch vertex index triples (`[u32; 3][n_patches]`).
    pub patch_vi_addr:   u64,
    /// Device address of the output pixel buffer (`u8[width * height * 3]`).
    pub pixels_addr:     u64,
    /// Device address of camera parameters (`f32[14]`): eye, fwd, right, up,
    /// tan_half_fov, aspect.
    pub camera_addr:     u64,
    /// Reinhard extended white-point (pre-computed on host as scene luminance
    /// maximum under Rec. 709 coefficients, clamped to at least 1.0).
    pub white:           f32,
    /// Output image width in pixels.
    pub width:           u32,
    /// Output image height in pixels.
    pub height:          u32,
    /// Number of patches.
    pub n_patches:       u32,
    /// Number of unique vertices in the vertex radiosity array.
    pub n_verts:         u32,
    /// Total hart count (from topology query).
    pub n_harts:         u32,
    /// Explicit padding to maintain size as a multiple of 8 bytes.
    pub _pad:            u32,
}

// SAFETY: `RenderArgs` is `#[repr(C)]`, contains no implicit padding, and
// every field is a fixed-width numeric type for which every bit pattern is a
// valid (if semantically meaningless) value, satisfying the `DeviceArgs`
// safety contract.
unsafe impl DeviceArgs for RenderArgs {}

// SAFETY: `PatchGeom` is `#[repr(C)]`, contains only `f32` fields (no pointers,
// no padding beyond the explicit `_p{0,1,2}` fields), and is valid for any bit pattern.
unsafe impl DevicePod for PatchGeom {}
