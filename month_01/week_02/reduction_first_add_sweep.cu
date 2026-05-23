// reduction_first_add_sweep.cu
// Day 8: Block-size sweep for reduction_first_add on RTX A5000 (sm_86).
// Halved block count, doubled per-thread work, min-of-10 timing.
//
// Build: nvcc -O2 -arch=sm_86 reduction_first_add_sweep.cu -o build/reduction_first_add_sweep

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do {                                            \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(_e));             \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)
__global__ void reduction_first_add(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     unsigned int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i    = blockIdx.x * blockDim.x + tid;
    unsigned int half = blockDim.x * gridDim.x;
    sdata[tid] = input[i] + input[i + half];
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 16; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid +  8];
        smem[tid] += smem[tid +  4];
        smem[tid] += smem[tid +  2];
        smem[tid] += smem[tid +  1];
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
static const int    A5000_MAX_THREADS_PER_SM = 1536;
static const int    A5000_MAX_BLOCKS_PER_SM  = 16;
static const int    A5000_WARP_SIZE          = 32;
static const double A5000_PEAK_BW_GBS        = 768.1;
static int max_blocks_per_sm(int blockSize) {
    int by_threads = A5000_MAX_THREADS_PER_SM / blockSize;
    int by_hwcap   = A5000_MAX_BLOCKS_PER_SM;
    return by_threads < by_hwcap ? by_threads : by_hwcap;
}
static double theoretical_occupancy_pct(int blockSize) {
    int blocks           = max_blocks_per_sm(blockSize);
    int resident_threads = blocks * blockSize;
    return (double)resident_threads / A5000_MAX_THREADS_PER_SM * 100.0;
}
int main() {
    const int    N     = 1 << 24;
    const size_t bytes = (size_t)N * sizeof(float);
    const double gb_moved = 1.0 * (double)bytes / 1.0e9;
    printf("reduction_first_add block-size sweep, N = %d (%.1f MB input)\n\n",
           N, bytes / (1024.0 * 1024.0));
    float* h_input  = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;
    float* d_input  = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes / 2));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    for (int w = 0; w < 5; ++w) {
        int bs = 256;
        int gs = N / (2 * bs);
        reduction_first_add<<<gs, bs, 2 * bs * sizeof(float)>>>(d_input, d_output, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    const int NUM_CONFIGS   = sizeof(block_sizes) / sizeof(block_sizes[0]);
    const int ITERS         = 10;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    printf("%-7s %-10s %-10s %-9s %-12s %-13s %-11s %s\n",
           "Block", "Grid", "Warps/Blk", "Blks/SM", "TheoryOcc%", "BestTime(ms)", "BW(GB/s)", "%Peak");
    printf("%-7s %-10s %-10s %-9s %-12s %-13s %-11s %s\n",
           "-----", "----", "---------", "-------", "----------", "------------", "--------", "-----");
    for (int b = 0; b < NUM_CONFIGS; ++b) {
        int bs = block_sizes[b];
        int gs = N / (2 * bs);
        int warps_per_block = (bs + A5000_WARP_SIZE - 1) / A5000_WARP_SIZE;
        int blks_per_sm     = max_blocks_per_sm(bs);
        double th_occ       = theoretical_occupancy_pct(bs);
        float best_ms = 1.0e9f;
        for (int iter = 0; iter < ITERS; ++iter) {
            CUDA_CHECK(cudaEventRecord(start));
            reduction_first_add<<<gs, bs, 2 * bs * sizeof(float)>>>(d_input, d_output, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms < best_ms) best_ms = ms;
        }
        double bw       = gb_moved / (best_ms / 1000.0);
        double pct_peak = bw / A5000_PEAK_BW_GBS * 100.0;
        int h_out_count = gs;
        float* h_output = (float*)malloc(h_out_count * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_output, d_output,
                              h_out_count * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float total = 0.0f;
        for (int j = 0; j < h_out_count; ++j) total += h_output[j];
        int correct = (fabsf(total - (float)N) < (float)N * 1e-3f) ? 1 : 0;
        free(h_output);
        printf("%-7d %-10d %-10d %-9d %-12.1f %-13.3f %-11.1f %-8.1f %s\n",
               bs, gs, warps_per_block, blks_per_sm, th_occ,
               best_ms, bw, pct_peak, correct ? "PASS" : "FAIL");
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    return 0;
}
