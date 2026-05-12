# RTX A5000 — Hardware Constants

Captured Day 1 from `month_01/week_01/build/device_query`.

| Parameter | Value | RTL/FPGA analog | Why it matters |
|---|---|---|---|
| Compute capability | 8.6 (Ampere GA102) | "ISA version" | Use `-arch=sm_86` for all kernels |
| SM count | 64 | Number of compute tiles | Outer parallelism floor — need >=64 blocks for full occupancy |
| Warp size | 32 | SIMT lane count | All block sizes should be multiples of 32 |
| Max threads / block | 1024 | Hard launch limit | Block sizes capped here |
| Max threads / SM | 1536 | Hyperthread cap | = 48 resident warps per SM |
| Max blocks / SM | 16 | | |
| Registers / SM | 65536 | Register file size | Occupancy constraint (regs/thread × threads <= 65536) |
| Registers / block | 65536 | | |
| Shared memory / block (default) | 48 KB | On-chip BRAM per tile | Tile-size budget for matmul / FA |
| Shared memory / SM | 100 KB | | Can opt-in beyond 48 KB at some occupancy cost |
| Total global memory | 23.68 GB GDDR6 | Off-chip DRAM | Largest tensor that fits |
| L2 cache | 6 MB | Shared LLC | Reused data across SMs lives here |
| Memory bus width | 384 bits | DRAM data bus | Bandwidth derivation: 2 × clock × bus/8 |
| Peak bandwidth | ~768 GB/s | DDR throughput | Roofline ceiling for bandwidth-bound kernels |
| Core clock | ~1.7 GHz | | |

## Quick derivations for kernel design

- **Resident warps per SM**: 1536 / 32 = **48 warps**.
- **Register pressure constraint**: with 64 regs/thread, max threads per SM = 65536 / 64 = 1024 (not 1536). Heavier kernels cap occupancy.
- **Shared memory pressure**: 1 block using 48 KB shared mem leaves room for only 1 block on that SM (since SM total is 100 KB and second block would need another 48 KB). Lower shared usage = more concurrent blocks per SM.
- **Roofline arithmetic intensity threshold**: peak FLOPS ÷ peak bandwidth. For Ampere A5000 FP32 ~27.8 TFLOPS / 768 GB/s = ~36 FLOPs/byte. Kernels below this are bandwidth-bound (optimize coalescing/caching); above are compute-bound (optimize tiling/register reuse).

