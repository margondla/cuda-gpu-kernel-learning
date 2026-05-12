// device_query.cu — Print the RTX A5000's hardware spec sheet.
// Smallest useful CUDA program: zero kernel launches, host-side queries only.

#include <cstdio>
#include <cuda_runtime.h>

// memoryClockRate / clockRate fields are deprecated in CUDA 12.x but still
// the cleanest way to print this info in a first program. Suppress to keep
// the compile silent; switch to cudaDeviceGetAttribute later if it matters.
#pragma nv_diag_suppress 1215

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    printf("CUDA-capable devices found: %d\n\n", count);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        err = cudaGetDeviceProperties(&p, i);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties(%d) failed: %s\n",
                    i, i, cudaGetErrorString(err));
            continue;
        }

        printf("=== Device %d: %s ===\n", i, p.name);
        printf("Compute capability        : %d.%d\n", p.major, p.minor);
        printf("Multiprocessor (SM) count : %d\n", p.multiProcessorCount);
        printf("Warp size                 : %d threads\n", p.warpSize);
        printf("Max threads per block     : %d\n", p.maxThreadsPerBlock);
        printf("Max threads per SM        : %d\n", p.maxThreadsPerMultiProcessor);
        printf("Max blocks per SM         : %d\n", p.maxBlocksPerMultiProcessor);
        printf("Max grid dim  (x, y, z)   : (%d, %d, %d)\n",
               p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        printf("Max block dim (x, y, z)   : (%d, %d, %d)\n",
               p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("Registers per block       : %d (32-bit)\n", p.regsPerBlock);
        printf("Registers per SM          : %d (32-bit)\n", p.regsPerMultiprocessor);
        printf("Shared memory per block   : %zu bytes (%zu KB)\n",
               p.sharedMemPerBlock, p.sharedMemPerBlock / 1024);
        printf("Shared memory per SM      : %zu bytes (%zu KB)\n",
               p.sharedMemPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024);
        printf("Total global memory       : %.2f GB\n",
               p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("L2 cache size             : %d bytes (%d KB)\n",
               p.l2CacheSize, p.l2CacheSize / 1024);
        printf("Memory bus width          : %d bits\n", p.memoryBusWidth);
        printf("Memory clock rate         : %d kHz (%.3f GHz effective)\n",
               p.memoryClockRate, p.memoryClockRate / 1.0e6);
        // Peak BW: factor of 2 for DDR (double data rate),
        // busWidth/8 to convert bits->bytes per transfer.
        printf("Peak memory bandwidth     : %.1f GB/s\n",
               2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6);
        printf("Core clock rate           : %d kHz (%.3f GHz)\n",
               p.clockRate, p.clockRate / 1.0e6);
        printf("ECC enabled               : %s\n", p.ECCEnabled ? "yes" : "no");
        printf("Concurrent kernels        : %s\n", p.concurrentKernels ? "yes" : "no");
        printf("Async engine count        : %d\n", p.asyncEngineCount);
        printf("\n");
    }
    return 0;
}
