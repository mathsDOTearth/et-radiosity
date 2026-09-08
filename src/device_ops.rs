//! ET-SoC-1 device operations: form-factor kernel launch and result retrieval.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use et_soc1::{Device, IoctlTransport, TraceConfig};
use et_soc1::trace::TraceBuffer;
use radiosity_abi::{FormFactorArgs, OccluderTri};

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

    // --- Upload occluder triangles ---
    // Convert host OccluderTriangle into the ABI OccluderTri layout.
    let occ_abi: Vec<OccluderTri> = occluders.iter().map(|o| OccluderTri {
        v0: o.v0, _pad0: 0.0,
        v1: o.v1, _pad1: 0.0,
        v2: o.v2, _pad2: 0.0,
    }).collect();
    let occ_bytes   = pod_as_bytes(&occ_abi);
    let (occ_addr, n_occluders) = if occ_abi.is_empty() {
        (0u64, 0u32)
    } else {
        let region = device.alloc(occ_bytes.len() as u64)
            .context("alloc occluder array")?;
        device.memcpy_h2d(occ_bytes, region.addr)
            .context("DMA occluder array")?;
        (region.addr, occ_abi.len() as u32)
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

    let ff: Vec<f32> = ff_raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

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
