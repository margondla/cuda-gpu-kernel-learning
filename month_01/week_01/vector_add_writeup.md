# Day 2: `vector_add` -- First Real CUDA Kernel

**Date**: May 13, 2026
**Hardware**: NVIDIA RTX A5000 (Ampere GA102, sm_86)
**Toolchain**: CUDA 12.8.93, nvcc release 12.8, Nsight Compute 2025.1.1.0
**Pod**: RunPod `envious_blush_cicada-migration` (`q7t5e54yaggy80`), US-IL-1 Secure Cloud

## Goal

Implement `C[i] = A[i] + B[i]` for `N = 2^24 = 16,777,216` floats. Embarrassingly parallel, no shared memory, no atomics. The point is not algorithmic complexity but practicing the full kernel workflow (write -> compile -> run -> verify -> profile -> analyze) and measuring achievable DRAM bandwidth on the A5000.

## Headline numbers

| Metric | Value |
|---|---|
| Kernel time (steady-state, min of 10) | **0.280 ms** |
| Effective bandwidth | **719 GB/s** |
| Peak DRAM bandwidth (A5000 spec) | 768.1 GB/s |
| **Fraction of peak** | **93.5%** |
| Correctness | PASS, 0 errors against CPU reference (1e-5 tolerance) |
| Arithmetic intensity | 0.083 FLOP/byte |

## Why this kernel is bandwidth-bound by design

The kernel performs **one FLOP per output element** (`A[i] + B[i]`) while moving **12 bytes** (read 4 bytes of A, read 4 bytes of B, write 4 bytes of C). Arithmetic intensity = 1/12 = **0.083 FLOP/byte**.

The A5000's roofline crossover from bandwidth-bound to compute-bound sits at AI ~ 36 FLOP/byte (27.8 TFLOPS / 768.1 GB/s). The kernel is **two orders of magnitude below the crossover**, so optimization analysis must focus on memory traffic, not arithmetic.

## Launch configuration and occupancy

For `N = 16,777,216` with block size 256: grid = 65,536 blocks; total threads = 16,777,216 (one per output element); 8 warps per block.

Theoretical occupancy on the A5000 by constraint:

| Constraint | Limit | Blocks/SM possible |
|---|---|---|
| Max threads/SM = 1,536 | 1,536 / 256 = 6 | **6 (binding)** |
| Max blocks/SM (hardware cap) | 16 | 16 |
| Register file: 65,536/SM, 16 effective regs x 256 threads = 4,096 regs/block | 65,536 / 4,096 = 16 | 16 |
| Shared memory (kernel uses 0 bytes) | unlimited | unbounded |

Binding constraint: **max threads/SM**. With 6 blocks x 256 threads = 1,536 = 100% of the SM's resident-thread capacity. **Theoretical occupancy = 100%.**

## Resource usage from compile

`nvcc --resource-usage -arch=sm_86 vector_add.cu` reports:

```
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 12 registers, used 0 barriers, 380 bytes cmem[0]
```

- **12 registers per thread** (rounded to 16 by sm_86 allocation granularity). Far below the 42-register threshold where register pressure starts cutting into occupancy.
- **Zero spills**: every live value stayed in registers.
- **Zero barriers**: no inter-thread coordination needed.
- **380 bytes cmem[0]**: kernel arguments plus per-launch metadata. Negligible.

## SASS observations

The kernel body compiles to 14 user-visible instructions:

```
S2R R6, SR_CTAID.X                                   ; blockIdx.x
S2R R3, SR_TID.X                                     ; threadIdx.x
IMAD R6, R6, c[0x0][0x0], R3                         ; i = blockIdx.x * blockDim.x + threadIdx.x
ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT           ; P0 = (i >= N)
@P0 EXIT                                             ; predicated exit (no branch)
MOV R7, 0x4
IMAD.WIDE R4, R6, R7, c[0x0][0x168]                  ; &A[i]
IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160]      ; &B[i] (with reuse hints)
LDG.E.CONSTANT R4, [R4.64]                           ; load A[i] via read-only cache
LDG.E.CONSTANT R3, [R2.64]                           ; load B[i] via read-only cache
IMAD.WIDE R6, R6, R7, c[0x0][0x170]                  ; &C[i] (scheduled in load shadow)
FADD R9, R4, R3                                      ; the entire arithmetic of the kernel
STG.E [R6.64], R9                                    ; store C[i]
EXIT
```

Four observations worth recording:
1. **`LDG.E.CONSTANT` instead of plain `LDG.E`.** The `const float* __restrict__` qualifiers on A and B unlocked the read-only L1 cache path. A concrete machine-code payoff for `__restrict__` -- not a theoretical optimization, a visible one.
2. **`.reuse` operand hints on the second `IMAD.WIDE`.** The compiler told the operand reuse cache to hold R6 and R7 across consecutive address computations, saving register-file ports. Ampere exposes a small per-warp operand reuse cache; the compiler is opting in.
3. **The C-address `IMAD.WIDE` is scheduled BETWEEN the two loads and the FADD.** Static instruction reordering by the compiler to hide load latency: the C-address compute has no data dependency on A or B's load results, so it can run while those loads are outstanding.
4. **A `ULDC.64 UR4, c[0x0][0x118]` (uniform datapath load) precedes the address-arithmetic block, not shown in the body listing above.** The kernel-argument base pointer is warp-invariant -- identical across all 32 threads in the warp -- so the compiler loaded it once into a uniform register via the uniform datapath rather than 32 times via the vector path. Ampere's uniform datapath is a separate silicon resource (separate issue port, separate register file) that runs alongside the vector ALUs without competing for them. Seeing `ULDC` here means the compiler correctly identified the warp-invariant computation. **Forward-looking for Week 1:** the tile base address in transpose kernels is also warp-invariant. Expect `ULDC` there too; if it's missing in the naive transpose, that's a code-gen smell worth investigating.

The 14-instruction body listing above counts vector-datapath ops only. The `MOV R1, c[0x0][0x28]` stack-frame prologue and the `ULDC.64` uniform load are excluded as architecturally separate (prologue runs once, uniform ops run on a different datapath).

The boundary check `if (i < N)` compiled to **predication, not branching**: `ISETP` sets a 1-bit predicate, `@P0 EXIT` is masked execution. For the last block where some threads have `i >= N`, the hardware masks those lanes rather than diverging the warp.

## Block-size sweep

Tested block sizes 32 through 1,024 with 5 warmup runs and min-of-10 timing per config (single binary, shared device state):

| Block | Grid | Warps/Blk | Blks/SM | TheoryOcc% | BestTime (ms) | BW (GB/s) | %Peak |
|---|---|---|---|---|---|---|---|
| 32 | 524288 | 1 | 16 | 33.3 | 0.426 | 472.7 | **61.5** |
| 64 | 262144 | 2 | 16 | 66.7 | 0.277 | 726.6 | **94.6** |
| 128 | 131072 | 4 | 12 | 100.0 | 0.280 | 719.7 | 93.7 |
| 256 | 65536 | 8 | 6 | 100.0 | 0.280 | 718.2 | 93.5 |
| 512 | 32768 | 16 | 3 | 100.0 | 0.279 | 721.0 | 93.9 |
| 1024 | 16384 | 32 | 1 | 66.7 | 0.285 | 707.5 | **92.1** |

Three observations:

1. **The plateau at block = 64 through 1024 sits within ~3% of itself**, averaging ~93.7% of peak. The kernel is at the practical bandwidth ceiling regardless of which of these block sizes is chosen.
2. **Block = 32 is the only meaningfully bad choice** at 61.5% of peak. With only 16 warps resident per SM (33% occupancy), the memory subsystem cannot stay saturated -- there aren't enough warps in flight to keep the DRAM request queue full when individual warps stall on loads.
3. **Block = 64 (67% occupancy) ties block = 256 (100% occupancy)** within measurement noise. This is the **Volkov "Better Performance at Lower Occupancy" effect**: for memory-bound kernels, the latency-hiding curve has a knee around 32 warps/SM rather than scaling linearly to 48.

## Methodology note: naive vs steady-state measurement

The simple `vector_add.cu` (run once, no warmup) reported **0.341 ms / 589.9 GB/s / 76.8% of peak**. The sweep with warmup and min-of-10 reported **0.280 ms / 718.2 GB/s / 93.5% of peak** for the same kernel at the same launch config.

The **17-percentage-point gap is entirely measurement methodology**, not the kernel. The naive measurement captures the GPU's clock-ramp from idle P-state to P0 boost during the kernel's execution. The sweep fires 5 warmup launches first, ensuring the GPU is at full boost before timing, then takes the minimum across 10 iterations to discard host-side jitter.

**Rule going forward**: warmup before measuring, take min-of-N for the headline number.

## What was NOT measured (and why)

The original plan included a Nsight Compute (`ncu --set full`) profile. This is **blocked on this pod** by NVIDIA's host-level `NVreg_RestrictProfilingToAdminUsers=1`, which RunPod's Secure Cloud sets for multi-tenant isolation. The error is `ERR_NVGPUCTRPERM`; no in-container privilege escalation can override a host-side driver flag.

The four alternative measurements above (resource usage, SASS, block-size sweep, methodology comparison) replace what `ncu` would have given us. Resolving the profiling situation is queued before the next kernel where it matters more (tiled matmul, Week 3).

## Lessons learned

1. **Methodology dominates apparent performance.** Always warmup, always min-of-N.
2. **Occupancy is necessary up to a knee, not a slope.** For memory-bound kernels on Ampere, the knee is near 32 warps/SM.
3. **`const` + `__restrict__` is a free optimization** with concrete SASS-level payoff (`LDG.E.CONSTANT`). Default-on for read-only inputs.
4. **The compiler does instruction-level latency hiding statically**, reordering address compute to fill load shadow.
5. **A first kernel hitting ~94% of peak DRAM is normal for a clean coalesced access pattern.** The remaining ~6% gap is fundamental memory subsystem overhead.

## Day 3 hand-off

- Resolve `ncu` profiling: either move to Community Cloud with a profiling-enabled host, or accept alternative measurements long-term.
- Next kernel candidate: **strided / transpose access pattern** to compare against today's coalesced baseline. Expected: bandwidth drop from ~94% to <30% of peak.