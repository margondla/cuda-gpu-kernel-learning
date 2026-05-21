// reduction_naive.cu — Mark Harris reduce0: interleaved addressing with divergent branching
// Pathologies by construction:
//   (a) warp divergence: at s=1 only even tids work, s=2 only tid%4==0, etc.
//   (b) bank conflicts: stride-s smem access pattern with s doubling each iter
// Goal: measure baseline; predict before compile; back-solve gap (Pitfall #21).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t e = call;                                              \
    if (e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(e));            \
        exit(1);                                                       \
    }                                                                  \
} while (0)

constexpr int N        = 1 << 24;   // 16,777,216 floats = 64 MB
constexpr int BLOCK    = 256;
constexpr int N_RUNS   = 20;
constexpr int N_WARMUP = 3;

__global__ void reduction_naive(const float* __restrict__ g_idata,
                                float* __restrict__ g_odata,
                                unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Interleaved addressing with divergent branch — Harris kernel #1
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float* h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_partial;
    int n_blocks = (N + BLOCK - 1) / BLOCK;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, n_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    size_t smem = BLOCK * sizeof(float);

    for (int i = 0; i < N_WARMUP; ++i) {
        reduction_naive<<<n_blocks, BLOCK, smem>>>(d_in, d_partial, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float min_ms = 1e30f;
    for (int run = 0; run < N_RUNS; ++run) {
        CUDA_CHECK(cudaEventRecord(start));
        reduction_naive<<<n_blocks, BLOCK, smem>>>(d_in, d_partial, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < min_ms) min_ms = ms;
    }

    float* h_partial = (float*)malloc(n_blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, n_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    double total = 0.0;
    for (int i = 0; i < n_blocks; ++i) total += h_partial[i];

    double bytes = (double)N * sizeof(float);
    double gbps  = bytes / (min_ms * 1.0e-3) / 1.0e9;
    double ratio_memcpy = gbps / 707.0;

    printf("=== reduction_naive (Harris reduce0) ===\n");
    printf("N            = %d (%.1f MB input)\n", N, bytes / 1.0e6);
    printf("Block        = %d, n_blocks = %d, smem/block = %zu B\n", BLOCK, n_blocks, smem);
    printf("Min time     = %.4f ms (min over %d runs, %d warmup)\n", min_ms, N_RUNS, N_WARMUP);
    printf("Throughput   = %.2f GB/s (N reads only; memcpy moves 2N)\n", gbps);
    printf("vs 707 GB/s  = %.1f%% of matched-N memcpy baseline\n", ratio_memcpy * 100.0);
    printf("Sum          = %.1f (expected %d, %s)\n",
           total, N, (total == (double)N) ? "OK" : "MISMATCH");

    free(h_in); free(h_partial);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}