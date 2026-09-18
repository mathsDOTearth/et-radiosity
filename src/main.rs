//! ET-SoC-1 radiosity renderer -- host binary.
//!
//! Loads a Cornell box scene from an OBJ file, tessellates the geometry into
//! surface patches, uploads the patches to the ET-SoC-1 via PCIe DMA, and
//! launches the form-factor kernel to compute the N x N form-factor matrix in
//! parallel across the device harts. The matrix is downloaded and a Jacobi
//! radiosity solve runs on the host CPU. The resulting per-patch radiance
//! values are ray-cast into a PNG image.
//!
//! # Usage
//!
//! ```text
//! et-radiosity --kernel ff-kernel.elf [OPTIONS]
//! ```
//!
//! See `--help` for all options.

mod device_ops;
mod render;
mod scene;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use et_soc1::{transport::IoctlTransport, Device};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// ET-SoC-1 radiosity renderer for the Cornell box.
///
/// Computes the form-factor matrix on the accelerator, solves radiosity on
/// the host, and saves a PNG image.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the ff-kernel ELF binary (built from ff-kernel/).
    #[arg(long, short = 'k')]
    kernel: PathBuf,

    /// Path to the Cornell box OBJ file.
    #[arg(long, short = 's', default_value = "assets/cornell_box.obj")]
    scene: PathBuf,

    /// Maximum patch side length, in metres. Controls tessellation density:
    /// smaller values increase patch count and solve accuracy at the cost of
    /// longer device kernel execution. 0.15 m is a practical default for the
    /// 1 m Cornell box; use 0.1 m for higher quality once kernel timing is
    /// established.
    #[arg(long, default_value_t = 0.15)]
    patch_size: f32,

    /// Number of Jacobi iterations for the radiosity solve.
    #[arg(long, default_value_t = 150)]
    iterations: u32,

    /// Output PNG width, in pixels.
    #[arg(long, default_value_t = 512)]
    width: u32,

    /// Output PNG height, in pixels.
    #[arg(long, default_value_t = 512)]
    height: u32,

    /// Output PNG file path.
    #[arg(long, short = 'o', default_value = "output.png")]
    output: PathBuf,

    /// ET-SoC-1 device index (as in /dev/etN_ops).
    #[arg(long, default_value_t = 0)]
    device: u32,

    /// Print device kernel trace messages to stderr. Allocates an additional
    /// trace buffer on the device.
    #[arg(long)]
    trace: bool,

    /// Reset all compute shires before launching any kernel. Use this to
    /// recover from a previous kernel crash that left the device in a faulted
    /// state. The device node remains open after the reset; DMA and host state
    /// are unaffected.
    #[arg(long)]
    reset: bool,

    /// Perform a full ETSOC hardware reset before launching any kernel. Closes
    /// the device node, triggers a firmware restart, and re-opens the device
    /// (may take up to 30 s). Use when --reset alone does not clear the fault.
    #[arg(long)]
    reset_device: bool,

    /// Override the compute-shire bitmask used for all kernel launches. Each
    /// set bit enables the corresponding shire (bit 0 = shire 0). The default
    /// is the full mask reported by the topology query. Use to exclude a faulty
    /// shire, e.g. --shire-mask 0xfffffffe skips shire 0.
    #[arg(long, value_parser = parse_hex_or_dec)]
    shire_mask: Option<u64>,

    /// Path to the render-kernel ELF binary (built from render-kernel/).
    /// When provided, ray-casting is performed on the ET-SoC-1 device rather
    /// than on the host CPU, and the render timing line reflects device time.
    #[arg(long, short = 'r')]
    render_kernel: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_hex_or_dec(s: &str) -> Result<u64, String> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else {
        s.parse::<u64>().map_err(|e| e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let args = Args::parse();

    // Wall-clock reference for total elapsed time.
    let t_start = Instant::now();

    // --- Load kernel ELF ---
    let kernel_elf = std::fs::read(&args.kernel)
        .with_context(|| format!("reading kernel ELF from {}", args.kernel.display()))?;
    eprintln!("Kernel: {} ({} bytes)", args.kernel.display(), kernel_elf.len());

    // --- Load and discretise scene ---
    let scene = scene::load_obj(&args.scene, args.patch_size)
        .with_context(|| format!("loading scene from {}", args.scene.display()))?;
    let n_emitters: usize = scene.patches.iter()
        .filter(|p| p.emission.iter().any(|&e| e > 0.0))
        .count();
    eprintln!("Scene:  {} patches, {} occluder triangles, {} emitting patches",
        scene.patches.len(), scene.occluders.len(), n_emitters);

    let t_after_load = Instant::now();

    // --- Open device ---
    let raw_device: Device<IoctlTransport> = Device::open(args.device)
        .with_context(|| format!("opening ET-SoC-1 device index {}", args.device))?;
    raw_device.set_default_launch_timeout(Duration::from_secs(300));

    // --- Optional soft shire reset ---
    // Recovers from a prior kernel crash that left one or more shires in a
    // faulted state. The device node stays open; firmware is not restarted.
    if args.reset {
        let topo = raw_device.topology().context("querying topology for reset")?;
        eprintln!("Resetting {} shires ({:#x})...", topo.num_shires(), topo.shire_mask);
        raw_device.reset_shires(topo.shire_mask).context("reset_shires")?;
        eprintln!("  shires reset -- OK");
    }

    // --- Optional full ETSOC hardware reset ---
    // Consumes the device, triggers a firmware restart, and re-opens the node.
    // Use when --reset alone does not unblock the device (up to 30 s).
    let device: Device<IoctlTransport> = if args.reset_device {
        eprintln!("Full ETSOC hardware reset (up to 30 s)...");
        let d = raw_device.reset_device().context("reset_device")?;
        d.set_default_launch_timeout(Duration::from_secs(300));
        eprintln!("  device re-opened -- OK");
        d
    } else {
        raw_device
    };

    // --- Resolve effective shire mask ---
    // The topology-reported mask is used by default; --shire-mask allows a
    // subset to be selected (e.g. to exclude a faulty shire for diagnostics).
    let effective_shire_mask: Option<u64> = args.shire_mask;

    // --- Compute form-factor matrix on device ---
    // compute_form_factors returns the matrix and the wall-clock duration of
    // the kernel launch itself (excluding DMA transfer time).
    let t_device_start = Instant::now();
    let (ff_matrix, kernel_elapsed) = device_ops::compute_form_factors(
        &device,
        &kernel_elf,
        &scene,
        args.trace,
        effective_shire_mask,
    )?;
    let device_elapsed = t_device_start.elapsed();

    // --- Solve radiosity on host ---
    let t_solve_start = Instant::now();
    let radiosities = device_ops::solve_radiosity(
        &scene.patches, &ff_matrix, args.iterations,
    );
    let solve_elapsed = t_solve_start.elapsed();

    // --- Render to PNG ---
    let t_render_start = Instant::now();
    let rk_kernel_elapsed: Option<Duration>;

    if let Some(rk_path) = &args.render_kernel {
        // Device-side ray-cast path.
        let rk_elf = std::fs::read(rk_path)
            .with_context(|| format!("reading render-kernel ELF from {}", rk_path.display()))?;
        eprintln!("Render kernel: {} ({} bytes)", rk_path.display(), rk_elf.len());

        let (pixels, rk_elapsed) = device_ops::render_on_device(
            &device,
            &rk_elf,
            &scene.patches,
            &radiosities,
            args.width,
            args.height,
            effective_shire_mask,
        )?;
        rk_kernel_elapsed = Some(rk_elapsed);

        let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
            args.width, args.height, pixels,
        ).expect("pixel buffer dimensions match image size");
        img.save(&args.output)
            .map_err(|e| anyhow::anyhow!("saving PNG: {e}"))?;
        eprintln!("  saved to {}", args.output.display());
    } else {
        // Host CPU ray-cast path.
        rk_kernel_elapsed = None;
        render::render_to_png(
            &scene.patches,
            &radiosities,
            args.width,
            args.height,
            &args.output,
        )?;
    }

    let render_elapsed = t_render_start.elapsed();
    let total_elapsed  = t_start.elapsed();

    // --- Timing summary ---
    // Columns: phase, seconds.
    // "On-card FF kernel" is the wall-clock time around launch_spmd for the
    // form-factor kernel (device compute + PCIe round-trip, excluding DMA).
    // "Device total" additionally includes DMA upload and download.
    // "On-card render kernel" (when present) is the analogous measurement for
    // the render kernel launch.
    eprintln!("\n=== Timing (seconds) ===");
    eprintln!("  Load (ELF + OBJ):     {:8.3}", t_after_load.duration_since(t_start).as_secs_f64());
    eprintln!("  Device total (FF):    {:8.3}  (DMA upload + kernel + DMA download)", device_elapsed.as_secs_f64());
    eprintln!("    On-card FF kernel:  {:8.3}  (launch_spmd wall clock)", kernel_elapsed.as_secs_f64());
    eprintln!("  Radiosity solve:      {:8.3}", solve_elapsed.as_secs_f64());
    if let Some(rke) = rk_kernel_elapsed {
        eprintln!("  Render (device):      {:8.3}  (DMA upload + kernel + DMA download)", render_elapsed.as_secs_f64());
        eprintln!("    On-card render:     {:8.3}  (launch_spmd wall clock)", rke.as_secs_f64());
    } else {
        eprintln!("  Render (host CPU):    {:8.3}", render_elapsed.as_secs_f64());
    }
    eprintln!("  -----");
    eprintln!("  Total:                {:8.3}", total_elapsed.as_secs_f64());

    Ok(())
}
