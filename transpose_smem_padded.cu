#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 4096
#define TILE 32
#define ITERATIONS 10

// Version B convention (matches transpose_naive): out[x*n+y] = in[y*n+x]
// Smem fixes the strided write by transposing the data within on-chip memory.
// Bank-conflict fires on the smem column-read tile[threadIdx.x][threadIdx.y].
__global__ void transpose_smem(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    __shared__ float tile[TILE][TILE+1];

    // Load phase: coalesced gmem read, row-write to smem (no bank conflict)
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }

    __syncthreads();

    // Store phase: block dims SWAPPED so threadIdx.x maps to output col.
    // Block (bx, by) writes output region rows [bx*TILE..) cols [by*TILE..).
    // Smem column-read serializes 32 ways on unpadded tile (the bank-conflict point).
    int x_out = blockIdx.x * TILE + threadIdx.y;
    int y_out = blockIdx.y * TILE + threadIdx.x;
    if (x_out < n && y_out < n) {
        out[x_out * n + y_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    transpose_smem<<<grid, block>>>(d_in, d_out, N); // warmup
    cudaDeviceSynchronize();;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float min_ms = 1e9f;
    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(start);
        transpose_smem<<<grid, block>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms < min_ms) min_ms = ms;
    }


    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("out[0]=%.0f out[1]=%.0f out[2]=%.0f out[3]=%.0f\n", h_out[0], h_out[1], h_out[2], h_out[3]);
    double gb = 2.0 * (double)N * N * sizeof(float) / 1e9;
    printf("min time: %.3f ms, bandwidth: %.1f GB/s\n", min_ms, gb / (min_ms / 1000.0));

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
