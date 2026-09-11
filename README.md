# et-radiosity

A proof-of-concept **radiosity renderer** targeting the
[Esperanto ET-SoC-1](https://www.esperanto.ai/product/) RISC-V many-core PCIe
accelerator card, written entirely in Rust.

The renderer loads a Cornell box scene from a Wavefront OBJ file, uploads the
geometry to the ET-SoC-1, computes the N x N form-factor matrix in parallel
across all available compute shires, solves the radiosity equation iteratively
on the host CPU, and saves the result as a PNG image.

---

## Architecture

```
et-radiosity/
+-- radiosity-abi/       no_std shared ABI types (Patch, OccluderTri, FormFactorArgs)
+-- ff-kernel/           RV64IMAC device kernel -- parallel form-factor computation
|   +-- src/main.rs
|   +-- link.ld          links at KERNEL_UMODE_ENTRY = 0x8005801000
|   +-- .cargo/config.toml
+-- src/                 x86-64 host binary
|   +-- main.rs          CLI, timing output, orchestration
|   +-- scene.rs         OBJ loader, patch tessellation, occluder extraction
|   +-- device_ops.rs    ET-SoC-1 upload, kernel launch, Jacobi solver
|   +-- render.rs        Moller-Trumbore ray caster, tone mapping, PNG output
+-- xtask/               cargo xtask helper (build, deploy, run)
+-- assets/
    +-- cornell_box.obj  room walls, ceiling light, white cube, blue sphere
    +-- cornell_box.mtl
```

### Crate stack

| Crate | Version | Role |
|-------|---------|------|
| `et-abi`  | 0.5 | Shared host/device ABI: `DeviceArgs` trait, arch constants |
| `et-rs`   | 0.5 | Host driver: PCIe DMA, kernel load/launch (`et_soc1` lib name) |
| `et-k-rs` | 0.5 | Device library: hart identity, `Grid`, scratchpad, trace (`et_kernel` lib name) |

### Algorithm

1. **Tessellation** -- each OBJ face is subdivided into sub-patches no larger
   than `--patch-size` metres. Non-wall faces (the cube and sphere) are also
   recorded as un-tessellated occluder triangles for the device shadow test.

2. **Form-factor computation (on device)** -- launched across all compute
   shires (up to 2048 harts on a fully populated ET-SoC-1). Each hart owns a
   disjoint row range of the N x N matrix. For each patch pair the kernel:

   - Performs a hemisphere filter (cosine-sign check on the unnormalised
     direction vector) to skip geometrically invalid pairs without a `sqrt`.
   - If the pair passes, casts a shadow ray using Moller-Trumbore intersection
     against the occluder triangle array.
   - If visible, applies the Nusselt-analogue form factor:

   ```
   F_ij = (cos theta_i * cos theta_j) / (pi * r^2) * A_j
   ```

   Each row is normalised after assembly to enforce energy conservation.

3. **Radiosity solve (on host)** -- Jacobi iteration over the downloaded
   form-factor matrix:

   ```
   B^(k+1)[i] = E[i] + rho[i] * sum_j ( F[i][j] * B^(k)[j] )
   ```

4. **Rendering** -- a perspective ray caster using exact Moller-Trumbore
   intersection against the per-patch triangle vertices assigns each pixel the
   radiosity of the nearest hit. Reinhard extended tone mapping and sRGB gamma
   correction are applied before PNG output.

---

## Prerequisites

### Local build machine

- Rust stable toolchain (edition 2024, rustc >= 1.88)
- `riscv64imac-unknown-none-elf` cross-compilation target:
  ```
  rustup target add riscv64imac-unknown-none-elf
  ```
- SSH and SCP access to the ET-SoC-1 test machine

### ET-SoC-1 test machine

- Esperanto PCIe kernel driver (`et_soc1` module, `/dev/et0_ops`)
- `et-rs` v0.5-compatible firmware on the card

---

## Manual build and deploy

This section covers every step individually. Use this during debugging or when
`cargo xtask` output is insufficient.

### Step 1 -- build the device kernel (cross-compile)

```sh
cd ff-kernel
cargo build --release
cd ..
```

Output: `ff-kernel/target/riscv64imac-unknown-none-elf/release/ff-kernel`

### Step 2 -- build the host binary

```sh
cargo build --release
```

Output: `target/release/et-radiosity`

> If cargo reports no change but the behaviour has not updated, force a rebuild:
> ```sh
> touch src/main.rs && cargo build --release
> ```

### Step 3 -- create the deployment directory on the test machine

```sh
ssh rich@aifoundry3 'mkdir -p ~/et-radiosity/assets'
```

### Step 4 -- copy binaries and assets

```sh
scp ff-kernel/target/riscv64imac-unknown-none-elf/release/ff-kernel \
    rich@aifoundry3:~/et-radiosity/ff-kernel.elf

scp target/release/et-radiosity \
    rich@aifoundry3:~/et-radiosity/et-radiosity

scp assets/cornell_box.obj assets/cornell_box.mtl \
    rich@aifoundry3:~/et-radiosity/assets/
```

### Step 5 -- restore the execute permission

`scp` does not preserve file permissions.

```sh
ssh rich@aifoundry3 'chmod +x ~/et-radiosity/et-radiosity'
```

### Step 6 -- run on the test machine

SSH into aifoundry3 for live stderr output:

```sh
ssh rich@aifoundry3

cd ~/et-radiosity

# Conservative starting point (~550 patches; completes in well under 1 s on device)
./et-radiosity \
    --kernel     ff-kernel.elf \
    --scene      assets/cornell_box.obj \
    --patch-size 0.3 \
    --iterations 50

# Intermediate quality
./et-radiosity \
    --kernel     ff-kernel.elf \
    --scene      assets/cornell_box.obj \
    --patch-size 0.15 \
    --iterations 150

# Full resolution (~2800 patches)
./et-radiosity \
    --kernel     ff-kernel.elf \
    --scene      assets/cornell_box.obj \
    --patch-size 0.1 \
    --iterations 150
```

Add `--trace` to any invocation to print per-hart diagnostic messages to
stderr via the ET-SoC-1 trace decoder.

### Step 7 -- retrieve the output image

From the local machine:

```sh
scp rich@aifoundry3:~/et-radiosity/output.png .
```

---

## Device recovery -- stuck kernel

If a kernel launch exceeds the driver's 10-second command-response timeout, the
Linux driver exits but the on-card RISC-V cores continue executing. Subsequent
invocations fail immediately at `load_kernel` because the firmware is ignoring
the submission queue.

`rmmod`/`modprobe` alone does **not** reset the firmware. A PCIe Fundamental
Level Reset is required.

```sh
# Identify the ET-SoC-1 PCI address (look for "Processing accelerators")
lspci | grep -i esperanto
# Example output: 02:00.0 Processing accelerators: Device 1e0a:eb01

# Issue the PCIe reset
echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/reset

# Reload the Linux driver (device nodes are removed by the reset)
sudo rmmod et_soc1 && sudo modprobe et_soc1

# Confirm the device is available
ls /dev/et*   # expect /dev/et0_mgmt and /dev/et0_ops
```

The host binary calls `Device::set_default_launch_timeout(Duration::from_secs(300))`
on startup (available from et-rs 0.5.3), which overrides the previously fixed
10-second deadline. With the hemisphere-filter optimisation and all shires active
(2048 harts), even the full 0.1 m resolution run completes in approximately
1 second on the device.

---

## cargo xtask (automated workflow)

For non-debugging use, the xtask crate automates the above steps.

```sh
# Build kernel only
cargo xtask kernel

# Build host binary only
cargo xtask host

# Build both
cargo xtask build

# Build, deploy, run, and retrieve output.png
cargo xtask run --remote rich@aifoundry3

# Deploy without running (then SSH in manually)
cargo xtask deploy --remote rich@aifoundry3

# Custom parameters
cargo xtask run \
    --remote     rich@aifoundry3 \
    --patch-size 0.1 \
    --iterations 200 \
    --width 1024 --height 1024 \
    --output     render.png
```

---

## Timing output

The host binary prints a timing summary to stderr after each run:

```
=== Timing (seconds) ===
  Load (ELF + OBJ):        0.031
  Device total:             0.847  (DMA upload + kernel + DMA download)
    On-card kernel:         0.812  (launch_spmd wall clock)
  Radiosity solve:          1.203
  Render:                   3.891
  -----
  Total:                    5.972
```

"On-card kernel" is the wall-clock duration of the `launch_spmd` call, which
includes device-side RISC-V compute and the PCIe command round-trip but
excludes DMA transfer time.

---

## Performance notes

With hemisphere filtering and all 32 compute shires (2048 harts), the
form-factor kernel time scales as O(N^2 / 2048) times the cost per surviving
pair. Approximately 30% of pairs pass the hemisphere filter in a Cornell box,
so shadow-ray traversal dominates only for the geometrically plausible subset.

Representative device kernel times on aifoundry3:

| `--patch-size` | Patches | Approx. kernel time |
|---------------|---------|---------------------|
| 0.3 m | ~550 | < 0.1 s |
| 0.2 m | ~1000 | ~0.15 s |
| 0.15 m | ~1250 | ~0.25 s |
| 0.1 m | ~2800 | ~1 s |

The host Jacobi solver and ray caster both run on the host CPU and are not
accelerated by the ET-SoC-1. At N = 2800 and 150 iterations the solver
performs ~1.2 billion multiply-accumulate operations; this dominates total
wall time at fine patch sizes.

---

## Known limitations and future work

- **Soft-float performance.** RV64IMAC lacks the F/D ISA extensions; all f32
  arithmetic in the device kernel is lowered to `compiler_builtins` routines.
  A future version could exploit the ET-SoC-1 Packed-Single instruction
  extensions (`FADD.PS`, `FMUL.PS`, `FSQRT.PS`) for 8-wide f32 SIMD throughput.

- **Host solve parallelism.** The Jacobi solver is single-threaded. Parallelising
  across host CPU cores with Rayon would reduce total wall time at large N.

- **Anti-aliasing.** The ray caster fires one primary ray per pixel.
  Multi-sample anti-aliasing would reduce aliasing at patch boundaries.

---

## Suggested et-rs API improvements

Building this project identified four gaps in the et-abi/et-rs API:

1. **`DevicePod` in `et-abi`** -- moving the marker trait from `et-rs` to
   `et-abi` would allow downstream crates to implement it for their own
   `#[repr(C)]` types without violating the orphan rule, removing the need for
   a raw `pod_as_bytes` unsafe workaround.

2. **`Device::upload_slice<T: DevicePod>()`** -- a combined `alloc` +
   `memcpy_h2d` convenience method eliminates three-call boilerplate for every
   array uploaded to device DRAM.

3. **`LaunchOptions::with_timeout(Duration)`** -- resolved in et-rs 0.5.3 via
   `Device::set_default_launch_timeout(Duration)` and `launch_spmd_opts` /
   `launch_spmd_traced_opts`. A per-launch override builder is a natural follow-on.

4. **`Device::reset_device()`** -- a host-callable firmware reset to recover
   from a stuck kernel without requiring a PCIe FLR or driver reload.

---

## References

- Goral, C. M., Torrance, K. E., Greenberg, D. P., & Battaile, B. (1984).
  Modelling the interaction of light between diffuse surfaces.
  *ACM SIGGRAPH 84 Proceedings*, 213-222.
- Moller, T. & Trumbore, B. (1997). Fast, minimum storage ray/triangle
  intersection. *Journal of Graphics Tools*, 2(1), 21-28.
- Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. (2002).
  Photographic tone reproduction for digital images.
  *ACM SIGGRAPH 2002 Proceedings*.
- Ward, G. J. (1994). The RADIANCE lighting simulation and rendering system.
  *ACM SIGGRAPH 94 Proceedings*.
