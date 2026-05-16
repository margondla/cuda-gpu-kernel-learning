# Transpose Kernel Performance Analysis

## 1. Summary

This writeup benchmarks three CUDA matrix transpose kernels on an RTX A5000 at N = 4096. Performance progresses from a naive coalesced-read / strided-write kernel at 182.8 GB/s (25.9% of memcpy), through an unpadded shared-memory kernel at 357.5 GB/s (50.6%), to a padded shared-memory kernel at 541.5 GB/s (76.6%). Shared-memory buffering is the dominant lever (1.96× over naive); padding to eliminate the 32-way smem bank conflict adds a further 1.51×. The remaining 23% gap to memcpy reflects `__syncthreads` cost, smem-access latency, and instruction overhead — not bank conflicts.

## 2. Bandwidth Results

| Kernel | Time (ms) | BW (GB/s) | % memcpy | × naive |
|---|---|---|---|---|
| memcpy | 0.181 | 707 | 100% | 3.87× |
| transpose_naive | 0.700 | 182.8 | 25.9% | 1.0× |
| transpose_smem (unpadded) | 0.375 | 357.5 | 50.6% | 1.96× |
| transpose_smem (padded) | 0.248 | 541.5 | 76.6% | 2.96× |

Bandwidth Results

| Kernel | Time (ms) | BW (GB/s) | % memcpy | × naive |
|---|---|---|---|---|
| memcpy | 0.181 | 707 | 100% | 3.87× |
| transpose_naive | 0.700 | 182.8 | 25.9% | 1.0× |
| transpose_smem (unpadded) | 0.375 | 357.5 | 50.6% | 1.96× |
| transpose_smem (padded) | 0.248 | 541.5 | 76.6% | 2.96× |

Bandwidth roughly doubles from naive to unpadded smem (1.96×) and climbs another ~50% from unpadded to padded (1.51×), landing the final kernel within striking distance of the memcpy ceiling. The multiplicative gains diverge, but the absolute contributions are nearly identical: smem buffering adds 174.7 GB/s, padding adds 184.0 GB/s — each intervention is worth roughly the same raw throughput, and they only look unequal because the second sits on a higher baseline. The residual 165.5 GB/s gap to memcpy reflects costs that padding cannot touch — the `__syncthreads` barrier, smem access latency on both load and store phases, and per-warp instruction overhead — all of which persist in any tile-based transpose regardless of bank layout.

## 3. Bank-Conflict Analysis

Shared memory on Ampere is striped across 32 banks that can service 32 distinct addresses in parallel per cycle, so a warp with one thread per bank completes its smem access in a single cycle. Route those same 32 threads onto a single bank at different addresses and the hardware serializes them into ~32 sequential cycles — the 32× cost penalty that makes the unpadded store phase the dominant overhead of that kernel.

### Bank distribution across access patterns

| Access pattern | t=0 | t=1 | t=2 | ... | t=31 | Distinct banks |
|---|---|---|---|---|---|---|
| Load `tile[ty][tx]` | 0 | 1 | 2 | ... | 31 | 32 → no conflict |
| Store `tile[tx][ty]` (unpadded `[32][32]`) | ty | ty | ty | ... | ty | 1 → 32-way conflict |
| Store `tile[tx][ty]` (padded `[32][33]`) | ty | (ty+1) mod 32 | (ty+2) mod 32 | ... | (ty+31) mod 32 | 32 → no conflict |

*e.g., for ty = 5: unpadded row = [5, 5, 5, ..., 5]; padded row = [5, 6, 7, ..., 4].*

### Derivation

For a row-major declared `__shared__ float tile[32][32]`, the address of element `tile[r][c]` is `base + (r*32 + c) * 4` bytes. Applying `bank(addr) = (addr / 4) mod 32` gives `bank(r,c) = (r*32 + c) mod 32 = c`, since `r*32` is always a multiple of 32. The bank depends only on the column index.

**Load phase**: the kernel writes `tile[threadIdx.y][threadIdx.x]`, so for a warp with `tx = threadIdx.x` varying 0..31 and `ty` fixed, the bank is `tx` — 32 distinct banks, no conflict. Completes in 1 smem cycle.

**Unpadded store phase**: the kernel reads `tile[threadIdx.x][threadIdx.y]`, so the bank is `ty` for every thread. All 32 threads target bank `ty`. The addresses are distinct (word offsets `ty, 32+ty, 64+ty, …, 992+ty`), so this is not a broadcast — broadcast requires identical addresses. Different addresses on the same bank → hardware serializes into ~32 sequential cycles. The store-phase read is ~32× slower than the symmetric-looking load-phase write.

**Padded store phase**: declaring `__shared__ float tile[32][33]` makes the row stride 33 words. The bank formula becomes `bank(r,c) = (r*33 + c) mod 32 = (r + c) mod 32`, since 33 mod 32 = 1. For the same access pattern `tile[tx][ty]`, the bank is `(tx + ty) mod 32` — a rotation across the warp that hits all 32 banks exactly once. No conflict.

## 4. Code-gen Observations

### Resource usage across kernels

| Kernel | Registers | Spills | smem (B) | cmem[0] (B) |
|---|---|---|---|---|
| `vector_add` | 12 | 0 | 0 | 376 |
| `transpose_naive` | 8 | 0 | 0 | 376 |
| `transpose_smem` (unpadded) | 12 | 0 | 4096 | 372 |
| `transpose_smem` (padded) | 12 | 0 | 4224 | 372 |

The +4 register delta from `transpose_naive` (8) to `transpose_smem` (12) reflects the additional index math required for the block-dimension swap in the store phase — the smem kernel computes both `(x, y)` and `(x_out, y_out)` pairs, which live simultaneously through the `__syncthreads` barrier. The +128 byte smem delta from unpadded (4096) to padded (4224) is one extra float per row × 32 rows, exactly the layout change introduced by `tile[TILE][TILE+1]`. No spills in any configuration — the compiler keeps all working values in registers even for the smem kernels despite the swap arithmetic.

### vector_add vs transpose_naive SASS

Two codegen divergences appear at SASS level. First, `vector_add` issues a `ULDC.64 UR4, c[0x0][0x160]` to load kernel-argument base pointers into a uniform register, reusing it across three pointer-arithmetic sites. `transpose_naive` does the opposite — it folds base addresses directly into each `IMAD.WIDE` as constant operands (`c[0x0][0x160]` and `c[0x0][0x168]`), bypassing the uniform-register load. The compiler chose this because `vector_add` has three pointers worth caching while `transpose_naive` has only two; folding is cheaper than load-and-reuse at that count. Second, `vector_add` carries `.reuse` operand hints on its `IMAD.WIDE` instructions, signaling the scheduler that the same register will feed the next op. `transpose_naive` has no `.reuse` hints — the compiler judged that downstream uses don't justify the hint slot.

### Padded vs unpadded transpose_smem SASS

Both SASS dumps come in at 112 lines, but `diff` reveals **three** changes, not one:

1. **Address-computation instruction**: unpadded uses `LEA R7, R7, R8, 0x5` (effective-address with shift-left-5, equivalent to multiply-by-32). Padded substitutes `IMAD R7, R7, 0x21, R8` (explicit multiply-by-33-plus-add). The role is equivalent — both compute a row-offset into the tile — but stride 32 admits the cheaper LEA-with-shift form because it's a power of 2; stride 33 forces an IMAD multiplication.
2. **Constant materialization**: unpadded uses `IMAD.MOV.U32 R3, RZ, RZ, 0x4` to set R3=4 (the float-size constant, paired with the LEA above); padded uses plain `MOV R3, 0x4`. Minor reorganization downstream of the LEA→IMAD swap.
3. **`IMAD.WIDE` immediate** shifts from `0x20` (32) to `0x21` (33) — the row-stride update for the smem store-address computation.

The takeaway: padding is not purely a layout change. Because `TILE+1 = 33` isn't a power of 2, the compiler loses the LEA-with-shift opportunity and pays for an extra IMAD multiply per address computation. The perf cost is negligible on Ampere — IMAD and LEA are both single-cycle — but the insight matters: **power-of-2 tile dimensions buy cheaper address arithmetic.**

## 5. Conclusions

The prediction-commitment exercise calibrated against measurement: a projected 610 GB/s (1.71× ratio) landed at 541.5 GB/s (1.51× ratio), revealing that the 32-way bank conflict accounted for **~65% of the unpadded smem overhead, not the 80% the prediction assumed**. The remaining 35% is `__syncthreads` barrier cost, smem-access latency, and per-warp instruction overhead — irreducible by bank-layout changes alone.

Two structural takeaways. First, **coalescing matters more than bank conflicts at this scale**: the naive→unpadded-smem step (which fixes the strided write via coalescing) delivered 1.96×, while unpadded→padded (which fixes the bank conflict) delivered 1.51×. If forced to pick one optimization, smem alone — even unpadded — is the bigger win. Second, the **residual 23% gap to memcpy is addressable but not via bank-layout changes**. Realistic next-step optimizations include vectorized loads (`LDG.E.128`), async smem pipelining via `cp.async` on Ampere, or larger tile sizes to amortize the `__syncthreads` cost across more useful work per block.
