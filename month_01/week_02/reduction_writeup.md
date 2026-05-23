# Reduction Kernel Writeup — Day 6 + Day 7

## Hardware + Environment
- GPU: NVIDIA RTX A5000 (sm_86), 24 GB VRAM
- Pod: RunPod Community Cloud, RTX A5000
- CUDA: 12.7 (driver 565.x)
- Compiler: nvcc -O2 -arch=sm_86
- Measurement: min of 20 runs, 3 warmup, cudaEvent timing

---

## Bandwidth Summary

| Kernel | Time (ms) | Throughput (GB/s) | % memcpy baseline |
|--------|-----------|-------------------|-------------------|
| memcpy baseline (N=4096, Secure Cloud) | — | 707 | 100% |
| reduce0 — naive (interleaved, divergent) | 0.4302 | 155.98 | 22.1% |
| reduce2 — sequential (tid < s, no divergence) | 0.2974 | 225.67 | 31.9% |
| reduce5 — unroll last warp | 0.1710 | 392.50 | 55.5% |

### Normalization caveat
Reduction reads N bytes (one pass over input). memcpy reads + writes 2N bytes.
All throughput figures use N bytes as the numerator — this is the correct normalization
for reduction, but it means the % memcpy column understates algorithmic efficiency
relative to a 2N-normalized baseline. An 80% figure here corresponds to ~40% of
peak DRAM bandwidth on a 2N basis.

### Measurement provenance
memcpy baseline is from Secure Cloud (Day 3, N=4096). All reduction kernels measured
on Community Cloud pod (GPU-0c6b4336, RTX A5000). reduce0 re-run on Community Cloud
confirmed 0.4302 ms vs Day 6 Secure Cloud 0.4300 ms — delta 0.05%, negligible.
Cross-environment comparison is clean.

---

## Kernel Analysis

### reduce0 — Harris reduce0 (interleaved addressing, divergent branch)

**Access pattern**: stride = 2*s, active thread condition `tid % (2*s) == 0`.
At s=1: active threads {0,2,4,...,254} — every other thread idle within each warp.
At s=2: active threads {0,4,8,...,252} — 3/4 idle. Divergence worsens each iteration.

**Warp divergence**: severe. The `tid % (2*s) == 0` guard splits warps into active
and idle lanes. Idle lanes stall but still consume warp slots, blocking the scheduler
from issuing other work. This extends wall-clock time beyond what a pure
bandwidth-bound model predicts.

**Bank conflicts**: none. Despite using shared memory, reduce0 has zero bank conflicts.
At stride s, active thread tid reads sdata[tid + s]. At s=1, active threads
{0,2,...,30} read sdata[{1,3,...,31}} — 16 distinct banks, no collision.
The stride-(2s) access pattern inherently avoids bank conflicts because active threads
are spaced far enough apart to land on unique banks.

SASS: BSSY/BSYNC visible — branch synchronization instructions confirming divergent
control flow materialized in hardware.

**Result**: 155.98 GB/s / 22.1% memcpy. Well below the 25-40% expected band —
likely due to re-entry calibration drift on Day 6 prediction (5.8x miss documented).

---

### reduce2 — Harris reduce2 (sequential addressing)

**Access pattern**: `if (tid < s) sdata[tid] += sdata[tid + s]`, s halving from
blockDim/2. Active threads are contiguous from tid=0 to tid=s-1.

**What changes vs reduce0**: two variables change simultaneously —
(1) warp divergence eliminated: contiguous active threads means no warp splitting.
(2) bank conflict pattern changes: tid and tid+s now index adjacent smem locations
at small s, which can produce 2-way conflicts. However at BLOCK=256, s>=1, the
access pattern is stride-1 at s=1 (tid reads sdata[tid+1]) — sequential, no conflict.

**Confound warning**: the 1.45x speedup over reduce0 cannot be cleanly attributed
to divergence elimination vs bank conflict pattern change — both variables change.
reduce1 (Harris reduce1, interleaved non-divergent with 2*s*tid indexing) would
isolate bank conflicts as the control experiment. reduce1 is a carry-forward item.

Reasoning from first principles: divergence elimination is the dominant contributor.
Bank conflicts in reduce0 were already zero (see above). The primary overhead removed
is warp serialization from divergent branches, not bank conflict resolution.
The 1.45x speedup is consistent with divergence being the dominant cost.

**Result**: 225.67 GB/s / 31.9% memcpy. 1.45x over reduce0.

---

### reduce5 — Harris reduce5 (unroll last warp)

**What changes vs reduce2**: final 6 iterations (s=32,16,8,4,2,1) unrolled in a
`__device__ warpReduce()` function using volatile smem pointer. No __syncthreads()
in the final phase — safe because s<=32 means only one warp remains active,
making warp-synchronous execution implicit.

**Why the gain is larger than expected**: naive scope analysis says "5 barriers removed
from the tail" and predicts modest gain. This underestimates two effects:
(1) __syncthreads() induces pipeline stalls that drain in-flight work — the cost
is not just the barrier instruction but the scheduler disruption it causes.
(2) Removing barriers from the tail changes the compiler's instruction scheduling
for those iterations, enabling better pipelining of the final phase. The cascading
effect on instruction throughput is larger than the direct barrier-removal cost.

Prediction was 0.27 ms / 35%; actual was 0.1710 ms / 55.5% — 1.74x miss.
The mental model had the cost of synchronization overhead inverted: the tail iterations
are cheap in arithmetic but expensive in scheduling overhead relative to their work.

**Remaining gap to 80-90%**: the global load phase is unchanged across all three kernels.
Each thread loads one element scalar — no vectorized loads (LDG.128), no ILP in
the load phase. The first reduction iteration (s=128) dominates memory bandwidth
consumption and is untouched. Closing the remaining ~25-35 percentage points requires
restructuring the load phase, not further synchronization cleanup.
Harris reduce6+ addresses this with "first add during load": each thread loads two
elements and adds them before writing to smem, halving block count and doubling
work per thread.

**Result**: 392.50 GB/s / 55.5% memcpy. 1.74x over reduce2, 2.52x over reduce0.

---

## Progression Pattern

reduce0 -> reduce2 -> reduce5 removes overhead in peeling order:
divergence (structural) -> synchronization barriers (scheduling) -> load phase (next).

Each optimization reveals the next bottleneck by removing the dominant overhead.
The easy structural fixes are exhausted after reduce5. What remains requires
touching the memory hierarchy directly.

---

## Prediction-Commitment Log

| Kernel | Predicted ms | Actual ms | Miss | Notes |
|--------|-------------|-----------|------|-------|
| reduce0 (Day 6) | 2.5 | 0.4300 | 5.8x | Re-entry after 4-day gap; calibration drift |
| reduce2 (Day 7) | 0.31 | 0.2974 | 4% | Direction correct; underestimated divergence latency cost |
| reduce5 (Day 7) | 0.27 | 0.1710 | 1.74x | Scope error: underweighted synchronization overhead in tail |

---

## Carry-Forward

- reduce1 (Harris reduce1): bank-conflict isolator. Needed to falsify the claim
  that divergence dominates the reduce0->reduce2 speedup. Skipped Day 7 for time.
- ncu hardware counters: ERR_NVGPUCTRPERM on both Secure Cloud and Community Cloud.
  Month 1 profiler deliverable drops to SASS + bandwidth + prediction-commitment.
  Dated gate 2026-05-24 passed — fallback is now permanent for Month 1.

---

## reduce1 — Harris reduce1 (interleaved, non-divergent) — Day 7 addendum

**Purpose**: control experiment to isolate bank conflict cost vs reduce2.
reduce1 fixes divergence (vs reduce0) but introduces bank conflicts via
`idx = 2*s*tid` strided smem access. reduce2 fixes both. Comparing the
three isolates each variable's contribution.

**Result**: 0.3106 ms / 216.05 GB/s / 30.6% memcpy

**Decomposition of reduce0 -> reduce2 speedup (1.45x)**:
- reduce0 -> reduce1: 0.4302 -> 0.3106 ms = 0.1196 ms saved (~91% of total gain)
  Cause: divergence elimination
- reduce1 -> reduce2: 0.3106 -> 0.2974 ms = 0.0132 ms saved (~9% of total gain)
  Cause: bank conflict elimination

**Falsification result**: the confound flagged in the original writeup is resolved.
Divergence elimination accounts for ~91% of the reduce0->reduce2 speedup.
Bank conflicts are a second-order effect in this kernel.

**Why bank conflicts are cheap here**: global memory bandwidth is the dominant
bottleneck. By the time s is small enough for interleaved addressing to create
severe conflicts, so little work remains that the absolute cost is negligible.
Bank conflicts matter when smem throughput is the bottleneck. It is not here.

**Updated cost model for sm_86 reductions**:
Warp divergence is ~9x more expensive than bank conflicts for this access pattern.
Prior assumption (bank conflicts clawback most of divergence gain) was wrong.
Calibrate future predictions accordingly.

**Prediction miss**: predicted 0.38 ms / 24.9%. Actual 0.3106 ms / 30.6%.
Error: overweighted bank conflict cost, underweighted divergence elimination value.
Direction of error: conservative on reduce1 performance (predicted worse than actual).

## Updated carry-forward

- reduce1 completed Day 7. Bank conflict falsification gate closed.
- ncu hardware counters: ERR_NVGPUCTRPERM permanent for Month 1.
- Next: reduce6 (first add during load) — attacks global load phase directly.

## reduce6 — First Add During Load (Harris reduce6)

### Result
- Time: 0.0973 ms | Throughput: 689.40 GB/s | 97.5% memcpy baseline
- vs reduce5: 0.1710 ms -> 0.0973 ms, 392.50 -> 689.40 GB/s, 1.76x speedup

### Structural change
Each thread loads 2 elements (g_in[i] + g_in[i + half]) before writing to smem.
Block count halved (65536 -> 32768). Work per thread doubled.

### SASS confirmation
0x00a0: LDG.E.CONSTANT R5, [R4.64]   // load element i
0x00b0: LDG.E.CONSTANT R6, [R6.64]   // load element i + N/2, back-to-back
0x00f0: FADD R2, R6, R5              // first add after both loads issued
0x0110: BAR.SYNC.DEFER_BLOCKING      // soft barrier, not hard stall

### Analytical conclusions
1. Load phase was the dominant bottleneck in reduce5. Two back-to-back LDGs
   with no intervening barrier allow the second load's latency to overlap with
   waiting on the first. This nearly saturated the memory bus (97.5% memcpy).
2. BAR.DEFER_BLOCKING issues without stalling -- warp only blocks if a
   dependent instruction is reached before the barrier completes. Hard BAR.SYNC
   would stall all threads immediately, destroying the ILP recovered by the
   two-load pattern.
3. The 42 percentage point jump (55.5% -> 97.5%) from one structural change
   confirms the bottleneck in reduce5 was not the reduction tree, not barriers,
   not unrolling -- it was the serial one-element-per-thread load pattern
   leaving the memory bus chronically underutilized.

### Prediction vs actual
Predicted: 0.105 ms / 640 GB/s / 90% | Actual: 0.0973 ms / 689 GB/s / 97.5%
Miss direction: underestimated ILP gain. Mental model assumed partial latency
recovery; actual result shows near-complete bus saturation from two loads alone.
