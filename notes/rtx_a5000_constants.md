# RTX A5000 Hardware Constants (Ampere GA102, CC 8.6)

Reference doc — re-read this whenever sizing a kernel. Source: `device_query.cu` output captured Day 1.

## Headline numbers

| Parameter | Value | Why it matters |
|---|---|---|
| Compute capability | 8.6 (Ampere GA102) | Use `-arch=sm_86` for all compiles |
| SM count | 64 | Need ≥64 blocks for full SM coverage |
| Warp size | 32 | All block sizes must be multiples of 32 |
| Max threads per block | 1024 | Hard launch limit |
| Max threads per SM | 1536 | = 48 resident warps per SM |
| Max blocks per SM | 16 | Block-count ceiling per SM |
| Registers per SM | 65536 (32-bit) | regs/thread × threads ≤ 65536 |
| Shared memory per block | 48 KB default, up to 100 KB opt-in | Tile-size budget for matmul/FA |
| Shared memory per SM | 100 KB | |
| Total global memory | 23.57 GB | Largest tensor that fits |
| L2 cache | 6 MB | Cross-SM reuse window |
| Memory bus width | 384 bits | |
| Peak memory bandwidth | 768.1 GB/s | Roofline ceiling for BW-bound kernels |
| Core clock | 1.695 GHz | |
| ECC | disabled | |
| Concurrent kernels | yes | Streams can overlap |
| Async copy engines | 2 | H2D and D2H can overlap with compute |

## Quick derivations

- **Resident warps per SM**: 1536 / 32 = **48 warps**.
- **Register-pressure ceiling**: with 64 regs/thread → 65536 / 64 = 1024 threads/SM max (not 1536). Heavy kernels are register-bound before they are thread-bound.
- **Shared-memory occupancy**: 48 KB/block × 2 = 96 KB fits under the 100 KB SM cap → **at most 2 blocks/SM** at default shared. Opt-in 100 KB shared locks to **1 block/SM**.
- **FP32 roofline crossover**: peak ~27.8 TFLOPS / 768 GB/s ≈ **36 FLOPs/byte**. Below → bandwidth-bound; above → compute-bound.

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
