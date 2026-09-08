//! Shared ABI types for the form-factor kernel.
//!
//! Both the host (`et-radiosity`) and the device kernel (`ff-kernel`) depend on
//! this crate, so it must compile in a `no_std` environment. All types are
//! `#[repr(C)]` and contain only POD fields, satisfying the requirements of the
//! [`et_abi::DeviceArgs`] safety contract.

#![no_std]

use et_abi::DeviceArgs;

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
/// offset 16: occluder_addr   -- device address of OccluderTri[n_occluders]
/// offset 24: n_patches
/// offset 28: n_harts
/// offset 32: n_occluders
/// offset 36: _pad
/// ```
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FormFactorArgs {
    /// Device address of the input patch array (`Patch[n_patches]`).
    pub patches_addr: u64,
    /// Device address of the output form-factor matrix (`f32[n_patches * n_patches]`),
    /// stored row-major: element `(i, j)` is at byte offset `(i * n_patches + j) * 4`.
    pub ff_matrix_addr: u64,
    /// Device address of the occluder triangle array (`OccluderTri[n_occluders]`).
    /// May be zero when `n_occluders == 0` (empty box, no visibility testing).
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
