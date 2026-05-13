// vector_add_sweep.cu
// Day 2 Step 11c: Block-size sweep on RTX A5000 (sm_86).
// Same kernel as vector_add.cu; tests block sizes 32 ... 1024 in one binary
// with shared device state, warmup runs, and min-of-10 timing.
//
// Build: nvcc -O2 -arch=sm_86 vector_add_sweep.cu -o build/vector_add_sweep

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
// Same kernel as the reference vector_add.cu
__global__ void vector_add(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// RTX A5000 (sm_86) hardware constants
static const int A5000_MAX_THREADS_PER_SM = 1536;
static const int A5000_MAX_BLOCKS_PER_SM  = 16;
static const int A5000_WARP_SIZE          = 32;
static const double A5000_PEAK_BW_GBS     = 768.1;

// Theoretical max blocks/SM ignoring register and shared-mem pressure (both ~0 here)
static int max_blocks_per_sm(int blockSize) {
    int by_threads = A5000_MAX_THREADS_PER_SM / blockSize;
    int by_hwcap   = A5000_MAX_BLOCKS_PER_SM;
    return by_threads < by_hwcap ? by_threads : by_hwcap;
}

static double theoretical_occupancy_pct(int blockSize) {
    int blocks = max_blocks_per_sm(blockSize);
    int resident_threads = blocks * blockSize;
    return (double)resident_threads / A5000_MAX_THREADS_PER_SM * 100.0;
}

int main() {
    const int N = 1 << 24;
    const size_t bytes = (size_t)N * sizeof(float);
    const double gb_moved = 3.0 * (double)bytes / 1.0e9;

    printf("vector_add block-size sweep, N = %d (%.1f MB per array, %.1f MB total H<->D)\n\n",
           N, bytes / (1024.0 * 1024.0), 3.0 * bytes / (1024.0 * 1024.0));

    // Host alloc + init
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // Device alloc + one-time H->D copy (data is read-only for the whole sweep)
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Warmup: drive GPU to P0 boost state before measurement
    for (int w = 0; w < 5; ++w) {
        vector_add<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Sweep configurations
    const int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    const int NUM_CONFIGS = sizeof(block_sizes) / sizeof(block_sizes[0]);
    const int ITERS = 10;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Table header
    printf("%-7s %-10s %-10s %-9s %-12s %-13s %-11s %s\n",
           "Block", "Grid", "Warps/Blk", "Blks/SM", "TheoryOcc%", "BestTime(ms)", "BW(GB/s)", "%Peak");
    printf("%-7s %-10s %-10s %-9s %-12s %-13s %-11s %s\n",
           "-----", "----", "---------", "-------", "----------", "------------", "--------", "-----");

    for (int b = 0; b < NUM_CONFIGS; ++b) {
        int bs = block_sizes[b];
        int gs = (N + bs - 1) / bs;
        int warps_per_block = (bs + A5000_WARP_SIZE - 1) / A5000_WARP_SIZE;
        int blks_per_sm = max_blocks_per_sm(bs);
        double th_occ = theoretical_occupancy_pct(bs);

        float best_ms = 1.0e9f;
        for (int iter = 0; iter < ITERS; ++iter) {
            CUDA_CHECK(cudaEventRecord(start));
            vector_add<<<gs, bs>>>(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms < best_ms) best_ms = ms;
        }

        double bw = gb_moved / (best_ms / 1000.0);
        double pct_peak = bw / A5000_PEAK_BW_GBS * 100.0;

        printf("%-7d %-10d %-10d %-9d %-12.1f %-13.3f %-11.1f %.1f\n",
               bs, gs, warps_per_block, blks_per_sm, th_occ, best_ms, bw, pct_peak);
    }

    // Correctness sanity check from the last run
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_C[i] - 3.0f) > 1e-5f) {
            if (errors < 3) printf("  mismatch at i=%d: got %f\n", i, h_C[i]);
            ++errors;
        }
    }
    printf("\ncorrectness (last config): %s (%d errors)\n",
           errors == 0 ? "PASS" : "FAIL", errors);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return errors == 0 ? 0 : 1;
}
