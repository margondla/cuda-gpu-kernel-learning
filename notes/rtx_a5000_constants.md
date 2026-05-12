# RTX A5000 Hardware Constants (Ampere GA102, CC 8.6)

Reference doc — re-read this whenever sizing a kernel. Source: `month_01/week_01/build/device_query` output captured Day 1.

## Headline numbers

| Parameter | Value | RTL/FPGA analog | Why it matters |
|---|---|---|---|
| Compute capability | 8.6 (Ampere GA102) | ISA version | Use `-arch=sm_86` for all compiles |
| SM count | 64 | Number of compute tiles | Need ≥64 blocks for full SM coverage |
| Warp size | 32 | SIMT lane count | All block sizes must be multiples of 32 |
| Max threads / block | 1024 | Hard launch limit | Block sizes capped here |
| Max threads / SM | 1536 | Hyperthread cap | = 48 resident warps per SM |
| Max blocks / SM | 16 | Block scheduler ceiling per tile | Block-count ceiling per SM |
| Registers / SM | 65536 (32-bit) | Register file size | regs/thread × threads ≤ 65536 |
| Shared memory / block | 48 KB default, up to 100 KB opt-in | On-chip BRAM per tile | Tile-size budget for matmul / FA |
| Shared memory / SM | 100 KB | On-chip BRAM per tile (total) | Cap shared across resident blocks |
| Total global memory | 23.57 GB GDDR6 | Off-chip DRAM | Largest tensor that fits |
| L2 cache | 6 MB | Shared LLC | Reused data across SMs lives here |
| Memory bus width | 384 bits | DRAM data bus | Bandwidth derivation: 2 × clock × bus / 8 |
| Peak memory bandwidth | 768.1 GB/s | DDR throughput | Roofline ceiling for BW-bound kernels |
| Core clock | 1.695 GHz | Pipeline clock | |
| ECC | disabled | Memory ECC parity | |
| Concurrent kernels | yes | Multi-stream kernel issue | Streams can overlap |
| Async copy engines | 2 | DMA channels | H2D and D2H can overlap with compute |

## Quick derivations for kernel design

- **Resident warps per SM**: 1536 / 32 = **48 warps**.
- **Register-pressure ceiling**: with 64 regs/thread → 65536 / 64 = 1024 threads/SM max (not 1536). Heavy kernels are register-bound before they are thread-bound.
- **Shared-memory occupancy**: 48 KB/block × 2 = 96 KB fits under the 100 KB SM cap → **at most 2 blocks/SM** at default shared. Opt-in 100 KB shared locks to **1 block/SM**.
- **FP32 roofline crossover**: peak ~27.8 TFLOPS / 768 GB/s ≈ **36 FLOPs/byte**. Below → bandwidth-bound (optimize coalescing/caching). Above → compute-bound (optimize tiling/register reuse).

## Compile flag (always)

    nvcc -O2 -arch=sm_86 <file>.cu -o build/<file>

Omitting `-arch=sm_86` triggers a deprecation warning and may default to a fallback target.

## Pre-launch checklist

1. Block size is a multiple of 32 (warp size).
2. Block size ≤ 1024 (hard limit).
3. regs/thread × threads/block × blocks/SM ≤ 65536 (register file).
4. shared/block × blocks/SM ≤ 100 KB (shared memory).
5. Grid size ≥ 64 blocks (cover all SMs).
6. Estimated arithmetic intensity compared against 36 FLOPs/byte — predict BW-bound vs compute-bound before profiling.

## Open data check

- `device_query.cu` reports 23.57 GB total global memory. `nvidia-smi --query-gpu=memory.total --format=csv` may report ~23.68 GB (full GDDR6 die capacity before driver-reserved regions). Reconcile on next session when profiling starts.
