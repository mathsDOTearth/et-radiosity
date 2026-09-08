//! Build, deploy, and run tasks for et-radiosity.
//!
//! Invoked as `cargo xtask <subcommand>` via the workspace `.cargo/config.toml`
//! alias. All paths are resolved relative to the workspace root so the task
//! works regardless of the working directory.
//!
//! # Subcommands
//!
//! ```text
//! cargo xtask kernel              -- cross-compile ff-kernel (riscv64imac)
//! cargo xtask host                -- compile et-radiosity (x86-64)
//! cargo xtask build               -- kernel then host
//! cargo xtask deploy --remote user@host [--dir ~/et-radiosity]
//! cargo xtask run    --remote user@host [options]
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "xtask", about = "Build and deploy tasks for et-radiosity")]
struct Cli {
    #[command(subcommand)]
    cmd: Task,
}

#[derive(Subcommand)]
enum Task {
    /// Cross-compile ff-kernel for riscv64imac-unknown-none-elf.
    Kernel,
    /// Compile the et-radiosity host binary for x86-64.
    Host,
    /// Build both the kernel and the host binary.
    Build,
    /// Deploy compiled binaries and scene assets to the ET-SoC-1 test machine via SCP.
    Deploy {
        /// SSH target for the test machine (e.g. user@et-testbox).
        #[arg(long)]
        remote: String,
        /// Destination directory on the test machine.
        #[arg(long, default_value = "~/et-radiosity")]
        dir: String,
    },
    /// Build, deploy, and run the renderer on the test machine, then retrieve output.png.
    Run {
        /// SSH target for the test machine (e.g. user@et-testbox).
        #[arg(long)]
        remote: String,
        /// Destination directory on the test machine.
        #[arg(long, default_value = "~/et-radiosity")]
        dir: String,
        /// Maximum patch side length in metres.
        #[arg(long, default_value_t = 0.1)]
        patch_size: f32,
        /// Number of radiosity Jacobi iterations.
        #[arg(long, default_value_t = 150)]
        iterations: u32,
        /// Output image width in pixels.
        #[arg(long, default_value_t = 512)]
        width: u32,
        /// Output image height in pixels.
        #[arg(long, default_value_t = 512)]
        height: u32,
        /// ET-SoC-1 device index (/dev/etN_ops).
        #[arg(long, default_value_t = 0)]
        device: u32,
        /// Enable device kernel trace output.
        #[arg(long)]
        trace: bool,
        /// Local path to write the retrieved PNG.
        #[arg(long, default_value = "output.png")]
        output: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = workspace_root()?;

    match cli.cmd {
        Task::Kernel          => build_kernel(&root),
        Task::Host            => build_host(&root),
        Task::Build           => { build_kernel(&root)?; build_host(&root) }
        Task::Deploy { remote, dir } => deploy(&root, &remote, &dir),
        Task::Run {
            remote, dir,
            patch_size, iterations, width, height, device, trace, output,
        } => {
            build_kernel(&root)?;
            build_host(&root)?;
            deploy(&root, &remote, &dir)?;
            run_remote(
                &remote, &dir,
                patch_size, iterations, width, height, device, trace,
            )?;
            retrieve_output(&remote, &dir, &output)
        }
    }
}

// ---------------------------------------------------------------------------
// Build steps
// ---------------------------------------------------------------------------

/// Cross-compiles ff-kernel for `riscv64imac-unknown-none-elf`.
///
/// The kernel crate carries its own `.cargo/config.toml` that selects the
/// target and passes the required linker flags; no extra arguments are needed.
fn build_kernel(root: &Path) -> Result<()> {
    eprintln!("--- Building ff-kernel (riscv64imac-unknown-none-elf) ---");
    let kernel_dir = root.join("ff-kernel");
    run(
        Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(&kernel_dir),
        "cargo build (ff-kernel)",
    )
}

/// Compiles the host binary for the current x86-64 toolchain.
fn build_host(root: &Path) -> Result<()> {
    eprintln!("--- Building et-radiosity (x86-64) ---");
    run(
        Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(root),
        "cargo build (host)",
    )
}

// ---------------------------------------------------------------------------
// Deploy step
// ---------------------------------------------------------------------------

/// Copies the kernel ELF, host binary, and scene assets to the test machine.
fn deploy(root: &Path, remote: &str, dir: &str) -> Result<()> {
    eprintln!("--- Deploying to {remote}:{dir} ---");

    let kernel_elf = root.join(
        "ff-kernel/target/riscv64imac-unknown-none-elf/release/ff-kernel",
    );
    let host_bin   = root.join("target/release/et-radiosity");
    let assets_dir = root.join("assets");

    // Ensure the remote directory structure exists.
    run(
        Command::new("ssh").args([remote, &format!("mkdir -p {dir}/assets")]),
        "ssh mkdir",
    )?;

    // SCP individual files. scp does not support globbing portably, so files
    // are copied one by one.
    scp(&kernel_elf,             &format!("{remote}:{dir}/ff-kernel.elf"))?;
    scp(&host_bin,               &format!("{remote}:{dir}/et-radiosity"))?;
    scp(&assets_dir.join("cornell_box.obj"), &format!("{remote}:{dir}/assets/cornell_box.obj"))?;
    scp(&assets_dir.join("cornell_box.mtl"), &format!("{remote}:{dir}/assets/cornell_box.mtl"))?;

    // scp does not preserve file permissions; set the execute bit explicitly.
    run(
        Command::new("ssh").args([
            remote,
            &format!("chmod +x {dir}/et-radiosity"),
        ]),
        "ssh chmod",
    )?;

    eprintln!("--- Deploy complete ---");
    Ok(())
}

// ---------------------------------------------------------------------------
// Run step
// ---------------------------------------------------------------------------

/// Executes the renderer on the remote machine via SSH.
#[allow(clippy::too_many_arguments)]
fn run_remote(
    remote:     &str,
    dir:        &str,
    patch_size: f32,
    iterations: u32,
    width:      u32,
    height:     u32,
    device:     u32,
    trace:      bool,
) -> Result<()> {
    eprintln!("--- Running on {remote} ---");

    let mut cmd_str = format!(
        "cd {dir} && ./et-radiosity \
            --kernel ff-kernel.elf \
            --scene  assets/cornell_box.obj \
            --patch-size {patch_size} \
            --iterations {iterations} \
            --width {width} \
            --height {height} \
            --output output.png \
            --device {device}",
    );
    if trace {
        cmd_str.push_str(" --trace");
    }

    run(Command::new("ssh").args([remote, &cmd_str]), "ssh run")
}

/// Retrieves `output.png` from the remote machine to a local path.
fn retrieve_output(remote: &str, dir: &str, local: &Path) -> Result<()> {
    eprintln!("--- Fetching output image ---");
    scp(&format!("{remote}:{dir}/output.png"), local)?;
    eprintln!("--- Image saved to {} ---", local.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Returns the workspace root directory.
///
/// Uses the `CARGO_MANIFEST_DIR` environment variable set by Cargo when
/// running the xtask, resolved to the parent (the workspace root).
fn workspace_root() -> Result<PathBuf> {
    // CARGO_MANIFEST_DIR points to the xtask package directory.
    let xtask_dir = std::env::var("CARGO_MANIFEST_DIR")
        .context("CARGO_MANIFEST_DIR not set; run via `cargo xtask`")?;
    Ok(PathBuf::from(xtask_dir)
        .parent()
        .context("xtask directory has no parent")?
        .to_owned())
}

/// Runs a [`Command`], streaming its output to the terminal, and returns an
/// error if the process exits with a non-zero status.
fn run(cmd: &mut Command, label: &str) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn `{label}`"))?;
    if status.success() {
        Ok(())
    } else {
        bail!("`{label}` exited with status {status}");
    }
}

/// Copies a single file or directory recursively via `scp`.
fn scp(src: impl AsRef<std::ffi::OsStr>, dst: impl AsRef<std::ffi::OsStr>) -> Result<()> {
    run(
        Command::new("scp").args([src.as_ref(), dst.as_ref()]),
        "scp",
    )
}
