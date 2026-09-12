//! ET-SoC-1 device operations: form-factor kernel launch and result retrieval.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use et_soc1::{Device, IoctlTransport, TraceConfig};
use et_soc1::trace::TraceBuffer;
use radiosity_abi::{FormFactorArgs, PatchGeom, RenderArgs};

use crate::render::build_vertex_radiosity;
use crate::scene::{Scene, ScenePatch};

const TRACE_BUFFER_BYTES: u64 = 4096 * 64;

// ---------------------------------------------------------------------------
// Form-factor matrix computation
// ---------------------------------------------------------------------------

/// Uploads scene data to the device, launches the form-factor kernel, and
/// returns the downloaded N x N form-factor matrix (row-major, `f32`) together
/// with the host-measured wall-clock duration of the `launch_spmd` call.
///
/// The returned [`Duration`] bounds the device-side compute time from below
/// (it includes the PCIe command round-trip but excludes DMA transfer time).
/// It is the most accurate host-observable proxy for on-card processing time.
///
/// When the scene contains interior occluder triangles the kernel performs a
/// Moller-Trumbore shadow test for every patch pair, producing correct binary
/// visibility and hence shadows in the radiosity solution.
pub fn compute_form_factors(
    device:     &Device<IoctlTransport>,
    kernel_elf: &[u8],
    scene:      &Scene,
    trace:      bool,
) -> Result<(Vec<f32>, Duration)> {
    let patches   = &scene.patches;
    let occluders = &scene.occluders;
    let n         = patches.len();

    eprintln!(
        "Device: {n}x{n} form-factor matrix, {} occluder triangles...",
        occluders.len()
    );

    let topo       = device.topology().context("querying topology")?;
    // Use all available compute shires, not just the first. The kernel-side
    // Grid::new(n_harts) uses the global hardware hart ID (CSR 0xCD0), so
    // hart IDs are unique across shires and row partitioning is correct.
    let shire_mask = topo.shire_mask;
    let n_harts    = topo.num_harts();
    eprintln!(
        "  {} shires ({shire_mask:#x}), {} harts total",
        topo.num_shires(), n_harts,
    );

    let kernel = device.load_kernel(kernel_elf).context("loading ff-kernel ELF")?;

    // --- Upload patch array ---
    let abi_patches: Vec<radiosity_abi::Patch> = patches.iter().map(|p| p.abi).collect();
    let patch_bytes = pod_as_bytes(&abi_patches);
    let patch_region = device.alloc(patch_bytes.len() as u64)
        .context("alloc patch array")?;
    device.memcpy_h2d(patch_bytes, patch_region.addr)
        .context("DMA patch array")?;

    // --- Upload occluder triangles in SoA layout ---
    //
    // The PS SIMD kernel processes 8 triangles per iteration using FLW.PS
    // (eight consecutive f32 values loaded into one 256-bit register). AoS
    // layout would require gather instructions; SoA permits simple strided
    // loads from contiguous memory.
    //
    // Layout: nine f32[n_padded] arrays -- v0x, v0y, v0z, v1x, v1y, v1z,
    // v2x, v2y, v2z -- where n_padded = ceil(n, 8). Trailing slots are 0.0.
    let n_occ = occluders.len();
    let (occ_addr, n_occluders) = if n_occ == 0 {
        (0u64, 0u32)
    } else {
        let n_padded = (n_occ + 7) & !7;  // round up to multiple of 8
        let mut soa = vec![0.0f32; 9 * n_padded];
        for (i, o) in occluders.iter().enumerate() {
            soa[0 * n_padded + i] = o.v0[0];
            soa[1 * n_padded + i] = o.v0[1];
            soa[2 * n_padded + i] = o.v0[2];
            soa[3 * n_padded + i] = o.v1[0];
            soa[4 * n_padded + i] = o.v1[1];
            soa[5 * n_padded + i] = o.v1[2];
            soa[6 * n_padded + i] = o.v2[0];
            soa[7 * n_padded + i] = o.v2[1];
            soa[8 * n_padded + i] = o.v2[2];
        }
        let soa_bytes = pod_as_bytes(&soa);
        let region = device.alloc(soa_bytes.len() as u64)
            .context("alloc SoA occluder array")?;
        device.memcpy_h2d(soa_bytes, region.addr)
            .context("DMA SoA occluder array")?;
        (region.addr, n_occ as u32)
    };

    // --- Allocate form-factor output ---
    let ff_bytes  = (n * n * 4) as u64;
    let ff_region = device.alloc(ff_bytes).context("alloc ff matrix")?;

    // --- Trace buffer (optional) ---
    let trace_region = if trace {
        Some(device.alloc(TRACE_BUFFER_BYTES).context("alloc trace buffer")?)
    } else {
        None
    };

    // --- Launch ---
    let args = FormFactorArgs {
        patches_addr:   patch_region.addr,
        ff_matrix_addr: ff_region.addr,
        occluder_addr:  occ_addr,
        n_patches:      n as u32,
        n_harts,
        n_occluders,
        _pad:           0,
    };

    // Time the kernel launch boundary. The call is synchronous: it returns
    // only once the device has finished execution and acknowledged completion.
    // This wall-clock duration therefore includes:
    //   (a) device-side RISC-V compute time (the dominant term), and
    //   (b) PCIe command/acknowledgement round-trip overhead.
    // It excludes the earlier DMA upload and the subsequent DMA download.
    let t_kernel = Instant::now();
    if let Some(tr) = &trace_region {
        device.launch_spmd_traced(
            &kernel, shire_mask, &args,
            TraceConfig::full(*tr, shire_mask),
        ).context("launch ff-kernel (traced)")?;
    } else {
        device.launch_spmd(&kernel, shire_mask, &args)
            .context("launch ff-kernel")?;
    }
    let kernel_elapsed = t_kernel.elapsed();

    // --- Print trace ---
    if let Some(tr) = &trace_region {
        let mut host_trace = vec![0u8; TRACE_BUFFER_BYTES as usize];
        device.memcpy_d2h(tr.addr, &mut host_trace)
            .context("download trace")?;
        if let Ok(tb) = TraceBuffer::parse(&host_trace) {
            for (hart, s) in tb.string_entries() {
                eprintln!("  [hart {hart}] {}", s.trim_end());
            }
        }
    }

    // --- Download form-factor matrix ---
    let mut ff_raw = vec![0u8; ff_bytes as usize];
    device.memcpy_d2h(ff_region.addr, &mut ff_raw)
        .context("download ff matrix")?;

    let mut ff: Vec<f32> = ff_raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Normalise each row so that sum_j F[i][j] <= 1 (energy conservation).
    // This is done on the host rather than in the kernel to avoid a store-buffer
    // ordering hazard: on the ET-Minion cores, reading device DRAM immediately
    // after writing it (without an explicit fence) may return stale content.
    // The host normalises using native f32 hardware after the DMA download,
    // and also sanitises any NaN/Inf values that could arise from near-zero
    // centroid distances in the point-to-point form-factor approximation.
    for row in ff.chunks_mut(n) {
        // Replace any non-finite values with 0 before summing.
        for v in row.iter_mut() {
            if !v.is_finite() { *v = 0.0; }
        }
        let row_sum: f32 = row.iter().sum();
        if row_sum > 1.0e-8 {
            for v in row.iter_mut() {
                *v /= row_sum;
            }
        }
    }

    eprintln!("  form-factor matrix downloaded ({} values)", ff.len());
    Ok((ff, kernel_elapsed))
}

// ---------------------------------------------------------------------------
// Radiosity solver (host CPU, Jacobi iteration)
// ---------------------------------------------------------------------------

/// Solves `B = E + rho * (F * B)` iteratively using the Jacobi method.
///
/// Returns per-patch RGB radiosity values.
pub fn solve_radiosity(
    patches:    &[ScenePatch],
    ff:         &[f32],
    iterations: u32,
) -> Vec<[f32; 3]> {
    let n = patches.len();
    assert_eq!(ff.len(), n * n);

    eprintln!("Host: Jacobi solve n={n}, {iterations} iterations...");

    let mut b:     Vec<[f32; 3]> = patches.iter().map(|p| p.emission).collect();
    let mut b_new: Vec<[f32; 3]> = vec![[0.0; 3]; n];

    for iter in 0..iterations {
        for i in 0..n {
            let rho       = patches[i].reflectance;
            let e         = patches[i].emission;
            let row_start = i * n;
            let mut gather = [0.0f32; 3];
            for j in 0..n {
                let ff_ij = ff[row_start + j];
                gather[0] += ff_ij * b[j][0];
                gather[1] += ff_ij * b[j][1];
                gather[2] += ff_ij * b[j][2];
            }
            b_new[i] = [
                e[0] + rho[0] * gather[0],
                e[1] + rho[1] * gather[1],
                e[2] + rho[2] * gather[2],
            ];
        }
        core::mem::swap(&mut b, &mut b_new);

        if iter % 20 == 0 {
            eprintln!("  iteration {iter}/{iterations}");
        }
    }

    eprintln!("  solve complete");
    b
}

// ---------------------------------------------------------------------------
// Render kernel launch
// ---------------------------------------------------------------------------

/// Uploads radiosity data to the device, launches the render kernel, and
/// returns the downloaded RGB pixel buffer (`width * height * 3` bytes,
/// row-major) together with the host-measured wall-clock duration of the
/// `launch_spmd` call.
///
/// The render kernel performs Moller-Trumbore ray-triangle intersection
/// against the patch geometry for every output pixel, applies
/// Gouraud-interpolated radiosity, Reinhard tone mapping, and sRGB gamma,
/// then writes the result to device DRAM. The host downloads the finished
/// pixel buffer after the kernel completes.
///
/// Camera convention matches the host-side `Camera::cornell_default`: eye at
/// (0.5, 0.5, -1.4), 45-degree vertical FOV, looking toward +Z.
pub fn render_on_device(
    device:      &Device<IoctlTransport>,
    kernel_elf:  &[u8],
    patches:     &[ScenePatch],
    radiosities: &[[f32; 3]],
    width:       u32,
    height:      u32,
) -> Result<(Vec<u8>, Duration)> {
    let n = patches.len();
    eprintln!(
        "Device render: {width}x{height}, {n} patches..."
    );

    let topo       = device.topology().context("querying topology")?;
    let shire_mask = topo.shire_mask;
    let n_harts    = topo.num_harts();
    eprintln!(
        "  {} shires ({shire_mask:#x}), {} harts total",
        topo.num_shires(), n_harts,
    );

    let kernel = device.load_kernel(kernel_elf)
        .context("loading render-kernel ELF")?;

    // --- Build Gouraud vertex radiosity (same algorithm as host render) ---
    let (vert_rad, patch_vi_usize) = build_vertex_radiosity(patches, radiosities);
    eprintln!("  {} unique vertices (Gouraud shading)", vert_rad.len());

    // Convert patch vertex indices from usize to u32 for device upload.
    let patch_vi: Vec<[u32; 3]> = patch_vi_usize
        .iter()
        .map(|&[a, b, c]| [a as u32, b as u32, c as u32])
        .collect();

    // --- Build patch geometry array ---
    let patch_geom: Vec<PatchGeom> = patches
        .iter()
        .map(|p| PatchGeom {
            v0: p.v0, _p0: 0.0,
            v1: p.v1, _p1: 0.0,
            v2: p.v2, _p2: 0.0,
        })
        .collect();

    // --- Upload patch geometry ---
    let geom_bytes  = pod_as_bytes(&patch_geom);
    let geom_region = device.alloc(geom_bytes.len() as u64)
        .context("alloc patch geometry array")?;
    device.memcpy_h2d(geom_bytes, geom_region.addr)
        .context("DMA patch geometry array")?;

    // --- Upload per-vertex radiosity ---
    let vrad_bytes  = pod_as_bytes(&vert_rad);
    let vrad_region = device.alloc(vrad_bytes.len() as u64)
        .context("alloc vertex radiosity array")?;
    device.memcpy_h2d(vrad_bytes, vrad_region.addr)
        .context("DMA vertex radiosity array")?;

    // --- Upload patch vertex index triples ---
    let vi_bytes  = pod_as_bytes(&patch_vi);
    let vi_region = device.alloc(vi_bytes.len() as u64)
        .context("alloc patch vertex index array")?;
    device.memcpy_h2d(vi_bytes, vi_region.addr)
        .context("DMA patch vertex index array")?;

    // --- Upload camera parameters ---
    // Layout: eye[3], fwd[3], right[3], up[3], tan_half_fov, aspect -- 14 f32.
    let aspect       = width as f32 / height as f32;
    let tan_half_fov = (22.5f32).to_radians().tan();
    let cam: [f32; 14] = [
        0.5, 0.5, -1.4,   // eye
        0.0, 0.0,  1.0,   // fwd
        1.0, 0.0,  0.0,   // right
        0.0, 1.0,  0.0,   // up
        tan_half_fov,
        aspect,
    ];
    let cam_bytes  = pod_as_bytes(&cam);
    let cam_region = device.alloc(cam_bytes.len() as u64)
        .context("alloc camera parameter buffer")?;
    device.memcpy_h2d(cam_bytes, cam_region.addr)
        .context("DMA camera parameters")?;

    // --- Compute white-point (Reinhard, Rec. 709 luminance) ---
    let white = radiosities
        .iter()
        .map(|&[r, g, b]| 0.2126 * r + 0.7152 * g + 0.0722 * b)
        .fold(0.0f32, f32::max)
        .max(1.0);

    // --- Allocate pixel output buffer ---
    let pixel_bytes  = (width as u64) * (height as u64) * 3;
    let pixel_region = device.alloc(pixel_bytes)
        .context("alloc pixel buffer")?;

    // --- Build and launch RenderArgs ---
    let args = RenderArgs {
        patch_geom_addr: geom_region.addr,
        vert_rad_addr:   vrad_region.addr,
        patch_vi_addr:   vi_region.addr,
        pixels_addr:     pixel_region.addr,
        camera_addr:     cam_region.addr,
        white,
        width,
        height,
        n_patches:       n as u32,
        n_verts:         vert_rad.len() as u32,
        n_harts,
        _pad:            0,
    };

    let t_kernel = Instant::now();
    device.launch_spmd(&kernel, shire_mask, &args)
        .context("launch render-kernel")?;
    let kernel_elapsed = t_kernel.elapsed();

    // --- Download pixel buffer ---
    let mut pixels_raw = vec![0u8; pixel_bytes as usize];
    device.memcpy_d2h(pixel_region.addr, &mut pixels_raw)
        .context("download pixel buffer")?;

    eprintln!("  pixel buffer downloaded ({} bytes)", pixels_raw.len());
    Ok((pixels_raw, kernel_elapsed))
}

// ---------------------------------------------------------------------------
// Utility: reinterpret a slice of POD structs as bytes
// ---------------------------------------------------------------------------

/// Reinterprets a slice of `#[repr(C)]` POD values as a byte slice for DMA.
///
/// # Safety
///
/// `T` must be a `#[repr(C)]` type with no padding and valid for any bit
/// pattern (plain-old-data). Callers ensure this through the type constraint.
fn pod_as_bytes<T: Copy>(data: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            data.as_ptr().cast::<u8>(),
            data.len() * core::mem::size_of::<T>(),
        )
    }
}
