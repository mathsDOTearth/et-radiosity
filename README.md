# et-radiosity

A proof-of-concept **radiosity renderer** targeting the
[Esperanto ET-SoC-1](https://www.esperanto.ai/product/) RISC-V many-core PCIe
accelerator card, written entirely in Rust.

The renderer loads a Cornell box scene from a Wavefront OBJ file, uses the
ET-SoC-1 to compute the form-factor matrix in parallel across up to 64 RISC-V
harts, solves the radiosity equation iteratively on the host CPU, and saves
the result as a PNG image.

---

## Architecture

```
et-radiosity/
+-- radiosity-abi/       no_std shared ABI (Patch, FormFactorArgs, DeviceArgs impl)
+-- ff-kernel/           RV64IMAC device kernel -- parallel form-factor computation
|   +-- src/main.rs
|   +-- link.ld
|   +-- .cargo/config.toml
+-- src/                 x86-64 host binary
|   +-- main.rs          CLI, orchestration
|   +-- scene.rs         OBJ loader, patch tessellation
|   +-- device_ops.rs    ET-SoC-1 kernel launch, radiosity Jacobi solver
|   +-- render.rs        perspective ray caster, PNG output
+-- assets/
    +-- cornell_box.obj
    +-- cornell_box.mtl
```

### Crate stack

| Crate | Version | Role |
|---|---|---|
| `et-abi` | 0.5 | Shared host/device ABI: `DeviceArgs` trait, arch constants |
| `et-rs`  | 0.5 | Host driver: PCIe DMA, kernel load/launch (`et_soc1` lib name) |
| `et-k-rs`| 0.5 | Device library: harts, scratchpad, trace (`et_kernel` lib name) |

### Algorithm

1. **Tessellation** -- each OBJ face is subdivided into sub-patches no larger
   than `--patch-size` metres. A typical Cornell box at 0.1 m produces
   approximately 500 patches.

2. **Form-factor computation (on device)** -- the form-factor kernel launches
   across all harts of the first shire. Each hart computes a contiguous block
   of rows of the N x N matrix using the Nusselt-analogue formula:

   ```
   F_ij = (cos theta_i * cos theta_j) / (pi * r^2) * A_j
   ```

   Visibility is unity for all patch pairs (exact for the convex Cornell box).
   Each row is normalised to enforce energy conservation.

3. **Radiosity solve (on host)** -- Jacobi iteration:

   ```
   B^(k+1)[i] = E[i] + rho[i] * sum_j ( F[i][j] * B^(k)[j] )
   ```

   Typically 100-200 iterations reach visual convergence for the Cornell box.

4. **Rendering** -- a software perspective ray caster assigns each pixel the
   radiosity of the nearest intersecting patch. Reinhard tone mapping and
   sRGB gamma correction are applied before PNG output.

---

## Prerequisites

### Local machine (build)

- Rust stable toolchain (edition 2024, rustc >= 1.88)
- `riscv64imac-unknown-none-elf` Rust target:
  ```
  rustup target add riscv64imac-unknown-none-elf
  ```
- LLVM linker (`rust-lld`, included with the Rust toolchain)
- SSH/SCP access to the ET-SoC-1 test machine

### Test machine (runtime)

- Esperanto SDK runtime (kernel module, `/dev/et0_ops`)
- `et-rs` v0.5 compatible firmware

---

## Build and run

All operations go through `cargo xtask`, defined in the `xtask/` crate and
aliased in `.cargo/config.toml`. No external build tools are required.

```sh
# Cross-compile ff-kernel (riscv64imac-unknown-none-elf):
cargo xtask kernel

# Compile the host binary (x86-64):
cargo xtask host

# Both in one step:
cargo xtask build
```

---

## Deploy and run

```sh
# Build, deploy, run, and retrieve output.png in one step:
cargo xtask run --remote user@et-testbox

# With custom parameters:
cargo xtask run \
    --remote user@et-testbox \
    --patch-size 0.05 \
    --iterations 200 \
    --width 1024 --height 1024 \
    --trace \
    --output render.png
```

To deploy without running (e.g. to run manually on the test machine):

```sh
cargo xtask deploy --remote user@et-testbox

# Then on the test machine:
./et-radiosity \
    --kernel ff-kernel.elf \
    --scene  assets/cornell_box.obj \
    --patch-size 0.1 \
    --iterations 150 \
    --width 512 \
    --height 512 \
    --output output.png \
    --trace
```

The `--trace` flag allocates a device trace buffer and prints per-hart
diagnostic messages to stderr via the ET-SoC-1 trace decoder.

---

## Performance notes

At 500 patches (0.1 m patch size), the form-factor kernel computes 250,000
form-factor evaluations. With 64 harts each handling approximately 8 rows of
500 evaluations, the device phase is dominated by soft-float `sqrt` calls
(the RV64IMAC ISA has no hardware F extension). The host Jacobi phase is
O(N^2) per iteration: at N=500 and 150 iterations this is ~37.5 M multiply-
accumulate operations, completing in well under a second on modern x86-64.

For larger scenes (N > 2000) the device benefit becomes more pronounced:
the form-factor computation scales as O(N^2 / P) per hart, while the host
solve scales as O(N^2 * iterations).

---

## Known limitations and future work

- **No inter-patch visibility testing.** The Nusselt formula assumes all patch
  pairs are mutually visible. This is exact for the Cornell box but incorrect
  for scenes with internal occluders. Future work: add a shadow-ray kernel
  (a separate et-k-rs kernel performing ray-AABB intersection per hart pair).

- **Single shire.** The renderer uses only `topo.first_shire()`. Extending to
  all shires would partition the row range across shires, giving linear scaling
  up to the full device hart count.

- **Soft-float performance.** RV64IMAC lacks the F/D ISA extensions, so all
  f32 arithmetic in the device kernel is lowered to `compiler_builtins`
  routines. A future version could use the ET-SoC-1 tensor extension
  (TensorFMA32) for the dot-product and multiply-accumulate steps.

- **Ray caster approximation.** The renderer uses a centroid-proximity test
  rather than exact triangle intersection, which may alias at patch boundaries.
  Replacing this with Moller-Trumbore triangle intersection (storing patch
  vertices on the host) would eliminate boundary artefacts.

- **Coplanar ceiling and light.** The OBJ defines a full ceiling quad and a
  separate smaller light quad at the same y coordinate. The ray caster resolves
  ties by returning the first intersection in patch order; in practice the light
  patch appears on top of the ceiling in the rendered image because it is listed
  last in the OBJ and tessellated after the ceiling. A production implementation
  would cut the light hole from the ceiling geometry.

---

## Reporting bugs

If you observe incorrect behaviour attributable to `et-abi`, `et-rs`, or
`et-k-rs` v0.5, please open an issue on the respective crate's repository with:

- Rust toolchain version (`rustc --version`)
- ET-SoC-1 firmware version
- Minimal reproducing example
- Expected vs. observed output

---

## References

- Goral, C. M., Torrance, K. E., Greenberg, D. P., & Battaile, B. (1984).
  Modelling the interaction of light between diffuse surfaces.
  *ACM SIGGRAPH 84 Proceedings*, 213-222.
- Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. (2002).
  Photographic tone reproduction for digital images.
  *ACM SIGGRAPH 2002 Proceedings*.
- Ward, G. J. (1994). The RADIANCE lighting simulation and rendering system.
  *ACM SIGGRAPH 94 Proceedings*.
