// transpose_naive.cu
// Day 4: naive transpose kernel
// out[x][y] = in[y][x]
// Coalesced reads (rows of in), strided writes (columns of out).
// Expected: 30-40% of memcpy_baseline (707 GB/s) = 210-283 GB/s at N=4096.

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while(0)

__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < N) {
        out[x * N + y] = in[y * N + x];
    }
}

int main() {
    const int N = 4096;
    const size_t bytes = (size_t)N * N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (size_t i = 0; i < (size_t)N * N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);

    // Warmup
    for (int i = 0; i < 3; i++) {
        transpose_naive<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float min_ms = 1e30f;
    for (int i = 0; i < 20; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        transpose_naive<<<grid, block>>>(d_in, d_out, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < min_ms) min_ms = ms;
    }

    double traffic_GB = 2.0 * bytes / 1e9;
    double bw = traffic_GB / (min_ms / 1000.0);
    printf("N=%d, min time = %.3f ms, bandwidth = %.1f GB/s\n", N, min_ms, bw);
    printf("Baseline memcpy at N=4096: 707 GB/s\n");
    printf("Fraction of baseline: %.1f%%\n", 100.0 * bw / 707.0);

    // Sanity check: transpose correctness on first 4 elements of output
    float h_check[4];
    CUDA_CHECK(cudaMemcpy(h_check, d_out, 4*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sanity: out[0]=%.0f (expect 0), out[1]=%.0f (expect %d), out[2]=%.0f (expect %d), out[3]=%.0f (expect %d)\n",
           h_check[0], h_check[1], N, h_check[2], 2*N, h_check[3], 3*N);

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
