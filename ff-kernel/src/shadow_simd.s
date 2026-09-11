# ET-SoC-1 Packed-Single SIMD shadow-ray batch tester.
#
# This file is assembled with the Esperanto GCC assembler
# (-march=rv64imaf_xet1p0), which recognises the PS extension instructions
# without the `aif.` prefix used by the Rust/LLVM toolchain.
#
# Function exposed:
#
#   uint64_t simd_shadow_test(
#       const float *soa,        // a0: SoA occluder base pointer
#       uint64_t     n_padded,   // a1: component-array stride, in f32 elements
#       uint64_t     full_batches,// a2: number of 8-wide batches to process
#       const float *ro,         // a3: ray origin  [3 x f32]
#       const float *rd,         // a4: ray direction [3 x f32]
#       const float *consts      // a5: [0.0, eps, T_MIN, dist-T_MIN] x f32[4]
#   );
#
#   Returns 0 if no triangle blocks the ray segment, nonzero otherwise.
#
# Algorithm: division-free Moller-Trumbore, 8 triangles per batch.
# All conditions are tested in sign-normalised form to avoid FDIV.PS (which
# traps to firmware on ET-SoC-1).  See ff-kernel/src/main.rs for details.
#
# Register map:
#   Integer:  a0=soa_p(advanced), a1=unused after prologue,
#             a2=batch counter, a3=&ro, a4=&rd, a5=&consts,
#             t0=stride_bytes, t1=tmp addr, t2=maskpopc result, t3=hit_accum
#   PS const: f25=0.0  f26=eps  f27=T_MIN  f28=dist-T_MIN
#             f29=rdx  f30=rdy  f31=rdz
#   PS work:  f0-f2   = v0x,v0y,v0z  (reused as sx,sy,sz)
#             f3-f5   = e1x,e1y,e1z
#             f6-f8   = e2x,e2y,e2z
#             f9-f11  = hx,hy,hz
#             f12     = a
#             f13-f14 = sign_pos, sign_neg (tmps)
#             f15     = sign_a
#             f16     = abs_a
#             f17     = hit  (per-batch accumulator)
#             f18     = scaled_u
#             f19-f21 = qx,qy,qz
#             f22     = scaled_v
#             f23     = scaled_t
#             f24     = tmp (booleans, ro broadcasts, etc.)

    .text
    .align 2
    .global simd_shadow_test
    .type   simd_shadow_test, @function

simd_shadow_test:
    # --------------- prologue -----------------------------------------------
    # No callee-saved PS registers to save: on RV64IMAC (no scalar F/D ext)
    # the standard ABI defines no float-register callee-save obligations.
    # All PS registers may be freely clobbered in a leaf function.

    # Bail out immediately if full_batches == 0.
    beqz    a2, .Lreturn_clear

    # stride_bytes = n_padded * 4
    slli    t0, a1, 2

    # hit_accum (integer, accumulates maskpopc results across all batches)
    li      t3, 0

    # Enable all 8 lanes: m0 = 0xFF.
    mov.m.x m0, zero, 0xff

    # Broadcast seven constants into PS registers f25-f31 (once per call).
    fbc.ps  f25, 0(a5)          # f25 = 0.0
    fbc.ps  f26, 4(a5)          # f26 = eps
    fbc.ps  f27, 8(a5)          # f27 = T_MIN
    fbc.ps  f28, 12(a5)         # f28 = dist - T_MIN
    fbc.ps  f29, 0(a4)          # f29 = rdx
    fbc.ps  f30, 4(a4)          # f30 = rdy
    fbc.ps  f31, 8(a4)          # f31 = rdz

    # --------------- batch loop ---------------------------------------------
.Lbatch:

    # --- Load 9 SoA components via walking address register (t1) ---
    flw.ps  f0, 0(a0)            # f0  = v0x[i..i+7]
    add     t1, a0, t0
    flw.ps  f1, 0(t1)            # f1  = v0y
    add     t1, t1, t0
    flw.ps  f2, 0(t1)            # f2  = v0z
    add     t1, t1, t0
    flw.ps  f3, 0(t1)            # f3  = v1x
    add     t1, t1, t0
    flw.ps  f4, 0(t1)            # f4  = v1y
    add     t1, t1, t0
    flw.ps  f5, 0(t1)            # f5  = v1z
    add     t1, t1, t0
    flw.ps  f6, 0(t1)            # f6  = v2x
    add     t1, t1, t0
    flw.ps  f7, 0(t1)            # f7  = v2y
    add     t1, t1, t0
    flw.ps  f8, 0(t1)            # f8  = v2z

    # --- e1 = v1 - v0 (overwrite f3-f5) ---
    fsub.ps f3, f3, f0           # e1x = v1x - v0x
    fsub.ps f4, f4, f1           # e1y
    fsub.ps f5, f5, f2           # e1z

    # --- e2 = v2 - v0 (overwrite f6-f8) ---
    fsub.ps f6, f6, f0           # e2x
    fsub.ps f7, f7, f1           # e2y
    fsub.ps f8, f8, f2           # e2z

    # --- h = cross(rd, e2) ---
    # hx = rdy*e2z - rdz*e2y  (fnmsub: fd = -(fs1*fs2) + fs3)
    fmul.ps  f9, f30, f8         # f9  = rdy * e2z
    fnmsub.ps f9, f31, f7, f9   # f9  = -(rdz*e2y) + f9  = hx
    # hy = rdz*e2x - rdx*e2z
    fmul.ps  f10, f31, f6        # f10 = rdz * e2x
    fnmsub.ps f10, f29, f8, f10  # f10 = -(rdx*e2z) + f10 = hy
    # hz = rdx*e2y - rdy*e2x
    fmul.ps  f11, f29, f7        # f11 = rdx * e2y
    fnmsub.ps f11, f30, f6, f11  # f11 = -(rdy*e2x) + f11 = hz

    # --- a = dot(e1, h) ---
    fmul.ps  f12, f3,  f9
    fmadd.ps f12, f4,  f10, f12
    fmadd.ps f12, f5,  f11, f12   # f12 = a

    # --- sign_a = sign(a) in {-1, 0, +1} ---
    flt.ps  f13, f25, f12         # f13: 1.0 where a > 0
    flt.ps  f14, f12, f25         # f14: 1.0 where a < 0
    fsub.ps f15, f13, f14         # f15 = sign_a

    # --- abs_a = a * sign_a ---
    fmul.ps f16, f12, f15         # f16 = |a|

    # --- hit = (|a| > eps): condition 1 ---
    flt.ps  f17, f26, f16         # f17: 1.0 where |a| > eps

    # --- s = ro - v0, using fbc.ps from stack (a3 = &ro) ---
    fbc.ps  f24, 0(a3)            # f24 = rox broadcast
    fsub.ps f0, f24, f0           # f0  = sx = rox - v0x
    fbc.ps  f24, 4(a3)            # f24 = roy
    fsub.ps f1, f24, f1           # f1  = sy
    fbc.ps  f24, 8(a3)            # f24 = roz
    fsub.ps f2, f24, f2           # f2  = sz

    # --- u_num = dot(s, h) -> scaled_u = u_num * sign_a (f18) ---
    fmul.ps  f18, f0, f9
    fmadd.ps f18, f1, f10, f18
    fmadd.ps f18, f2, f11, f18   # f18 = u_num
    fmul.ps  f18, f18, f15       # f18 = scaled_u

    # --- cond2a: scaled_u >= 0  (fle: 0 <= scaled_u) ---
    fle.ps  f24, f25, f18
    fmul.ps f17, f17, f24

    # --- cond2b: scaled_u <= abs_a ---
    fle.ps  f24, f18, f16
    fmul.ps f17, f17, f24

    # --- q = cross(s, e1) ---
    # qx = sy*e1z - sz*e1y
    fmul.ps  f19, f1, f5
    fnmsub.ps f19, f2, f4, f19   # qx
    # qy = sz*e1x - sx*e1z
    fmul.ps  f20, f2, f3
    fnmsub.ps f20, f0, f5, f20   # qy
    # qz = sx*e1y - sy*e1x
    fmul.ps  f21, f0, f4
    fnmsub.ps f21, f1, f3, f21   # qz

    # --- v_num = dot(rd, q) -> scaled_v = v_num * sign_a (f22) ---
    fmul.ps  f22, f29, f19
    fmadd.ps f22, f30, f20, f22
    fmadd.ps f22, f31, f21, f22  # f22 = v_num
    fmul.ps  f22, f22, f15       # f22 = scaled_v

    # --- cond3a: scaled_v >= 0 ---
    fle.ps  f24, f25, f22
    fmul.ps f17, f17, f24

    # --- cond3b: scaled_u + scaled_v <= abs_a ---
    fadd.ps f24, f18, f22        # uv_sum
    fle.ps  f24, f24, f16
    fmul.ps f17, f17, f24

    # --- t_num = dot(e2, q) -> scaled_t = t_num * sign_a (f23) ---
    fmul.ps  f23, f6, f19
    fmadd.ps f23, f7, f20, f23
    fmadd.ps f23, f8, f21, f23   # f23 = t_num
    fmul.ps  f23, f23, f15       # f23 = scaled_t

    # --- cond4a: scaled_t > T_MIN * abs_a ---
    fmul.ps f24, f27, f16        # T_MIN * abs_a
    flt.ps  f24, f24, f23        # 1.0 where scaled_t > T_MIN * abs_a
    fmul.ps f17, f17, f24

    # --- cond4b: scaled_t < (dist-T_MIN) * abs_a ---
    fmul.ps f24, f28, f16        # (dist-T_MIN) * abs_a
    flt.ps  f24, f23, f24        # 1.0 where scaled_t < (dist-T_MIN)*abs_a
    fmul.ps f17, f17, f24

    # --- extract result: m1[i] = 1 if hit[i] > 0 ---
    fltm.ps m1, f25, f17         # m1 bit i set where hit[i] > 0
    maskpopc t2, m1              # t2 = count of hit triangles in this batch
    or      t3, t3, t2           # accumulate across all batches

    # Early exit: if any hit found, stop immediately.
    bnez    t3, .Lreturn_hit

    # Advance soa_p by 32 bytes (8 f32), decrement counter.
    addi    a0, a0, 32
    addi    a2, a2, -1
    bnez    a2, .Lbatch

.Lreturn_clear:
    li      a0, 0
    ret

.Lreturn_hit:
    mv      a0, t3
    ret

    .size simd_shadow_test, . - simd_shadow_test
