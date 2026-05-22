#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

// Harris reduce1: interleaved addressing, non-divergent
// All threads execute same code path (no warp divergence)
// BUT: idx = 2*s*tid creates strided smem access -> bank conflicts
// Control experiment: isolates bank conflict cost vs reduce2
__global__ void reduction_non_divergent(const float* __restrict__ g_in,
                                        float* __restrict__ g_out,
                                        unsigned int n)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_in[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        int idx = 2 * s * tid;
        if (idx + s < blockDim.x)
            sdata[idx] += sdata[idx + s];
        __syncthreads();
    }

    if (tid == 0) g_out[blockIdx.x] = sdata[0];
}

int main(void)
{
    const int   N      = 1 << 24;
    const int   BLOCK  = 256;
    const int   GRID   = (N + BLOCK - 1) / BLOCK;
    const int   WARMUP = 3;
    const int   RUNS   = 20;
    const float bytes  = (float)N * sizeof(float);

    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N    * sizeof(float));
    cudaMalloc(&d_out, GRID * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP; i++)
        reduction_non_divergent<<<GRID, BLOCK, BLOCK*sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    float min_ms = FLT_MAX;
    for (int i = 0; i < RUNS; i++) {
        cudaEventRecord(start);
        reduction_non_divergent<<<GRID, BLOCK, BLOCK*sizeof(float)>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        if (ms < min_ms) min_ms = ms;
    }

    float *h_out = (float*)malloc(GRID * sizeof(float));
    cudaMemcpy(h_out, d_out, GRID * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0.0f;
    for (int i = 0; i < GRID; i++) sum += h_out[i];

    float gb_per_s = (bytes / (min_ms * 1e-3f)) / 1e9f;
    float memcpy_baseline = 707.0f;

    printf("=== reduction_non_divergent (Harris reduce1) ===\n");
    printf("N            = %d (%.1f MB input)\n", N, bytes/1e6f);
    printf("Block        = %d, n_blocks = %d, smem/block = %lu B\n",
           BLOCK, GRID, BLOCK*sizeof(float));
    printf("Min time     = %.4f ms (min over %d runs, %d warmup)\n", min_ms, RUNS, WARMUP);
    printf("Throughput   = %.2f GB/s (N reads only; memcpy moves 2N)\n", gb_per_s);
    printf("vs 707 GB/s  = %.1f%% of matched-N memcpy baseline\n", 100.f*gb_per_s/memcpy_baseline);
    printf("Sum          = %.1f (expected %d, %s)\n",
           sum, N, fabsf(sum - N) < 1.0f ? "OK" : "WRONG");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
