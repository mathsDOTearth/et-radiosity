//! Build script for render-kernel.
//!
//! Optionally assembles `src/nearest_hit_simd.s` using the Esperanto GCC
//! toolchain assembler (`riscv64-unknown-elf-as -march=rv64imaf_xet1p0`),
//! which recognises the ET-SoC-1 Packed-Single (PS) extension instructions.
//! The assembly source is not yet present at initial crate creation; assembly
//! is skipped when the file does not exist, so the kernel compiles cleanly
//! from the outset.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Only assemble for the RISC-V target; skip on the host (e.g. `cargo
    // check` on x86) where the ET assembler would produce an incompatible ABI.
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.starts_with("riscv") {
        return;
    }

    // Re-run if the assembly source is added or modified.
    println!("cargo:rerun-if-changed=src/nearest_hit_simd.s");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest_dir.join("src").join("nearest_hit_simd.s");

    // Assembly source is optional: return early if not yet written.
    if std::fs::metadata(&src).is_err() {
        return;
    }

    let et_as = "/opt/et/bin/riscv64-unknown-elf-as";
    let et_ar = "/opt/et/bin/riscv64-unknown-elf-ar";

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out_dir.join("nearest_hit_simd.o");
    let lib = out_dir.join("libnearest_hit_simd.a");

    // Assemble.
    let status = Command::new(et_as)
        .args([
            "-march=rv64imaf_xet1p0",
            src.to_str().unwrap(),
            "-o",
            obj.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run {et_as}: {e}"));
    assert!(status.success(), "assembler exited with {status}");

    // Archive the object file into a static library that Cargo can link.
    let status = Command::new(et_ar)
        .args([
            "crs",
            lib.to_str().unwrap(),
            obj.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to run {et_ar}: {e}"));
    assert!(status.success(), "ar exited with {status}");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=nearest_hit_simd");
}
