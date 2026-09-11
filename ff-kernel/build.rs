//! Build script for ff-kernel.
//!
//! Assembles `src/shadow_simd.s` using the Esperanto GCC toolchain assembler
//! (`riscv64-unknown-elf-as -march=rv64imaf_xet1p0`), which recognises the
//! ET-SoC-1 Packed-Single (PS) extension instructions.  The resulting object
//! file is archived and linked into the kernel ELF by Cargo.
//!
//! The Rust/LLVM integrated assembler (nightly-only `+xaifet`) is not used
//! because only the stable toolchain is installed on the build machine.

use std::process::Command;
use std::path::PathBuf;

fn main() {
    // Only assemble for the RISC-V target; skip on the host (e.g. `cargo
    // check` on x86) where the ET assembler would produce an incompatible ABI.
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.starts_with("riscv") {
        return;
    }

    let et_as = "/opt/et/bin/riscv64-unknown-elf-as";
    let et_ar = "/opt/et/bin/riscv64-unknown-elf-ar";

    // Locate the source assembly file relative to this build script.
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest_dir.join("src").join("shadow_simd.s");

    // Output directory provided by Cargo.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out_dir.join("shadow_simd.o");
    let lib = out_dir.join("libshadow_simd.a");

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

    // Tell Cargo where to find the library.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=shadow_simd");

    // Re-run this build script if the assembly source changes.
    println!("cargo:rerun-if-changed=src/shadow_simd.s");
}
