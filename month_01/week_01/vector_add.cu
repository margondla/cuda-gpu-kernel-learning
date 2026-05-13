// vector_add.cu
// Day 2: first real CUDA kernel
// Computes C[i] = A[i] + B[i] for N = 2^24 floats on RTX A5000 (sm_86)
//
// Build:   nvcc -O2 -arch=sm_86 vector_add.cu -o build/vector_add
// Profile: ncu --set full -o ../../profiles/vector_add ./build/vector_add

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// CUDA_CHECK wraps every CUDA API call. CUDA errors do not throw, do not
// crash; they silently return cudaError_t. Unwrapped calls = silent bugs.
// This is the single most important habit to form in CUDA programming.
#define CUDA_CHECK(call) do {                                            \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(_e));             \
        exit(EXIT_FAILURE);                                              \
    }                                                                    \
} while (0)

// The kernel itself.
// __global__  = called from host, executes on device, returns void.
// Each thread computes exactly one output element.
__global__ void vector_add(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N) {
    // Global thread index = (which block am I in) * (block size) + (my lane)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard the tail: last block has spillover threads when N % blockDim != 0
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 24;                  // 16,777,216 elements
    const size_t bytes = (size_t)N * sizeof(float);

    printf("vector_add: N = %d (%.1f MB per array, %.1f MB total H<->D)\n",
           N, bytes / (1024.0 * 1024.0), 3.0 * bytes / (1024.0 * 1024.0));

    // --- Phase 1+2: host alloc + init ---
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // --- Phase 3: device alloc ---
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // --- Phase 4: H -> D copy ---
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // --- Phase 5: launch ---
    const int blockSize = 256;
    const int gridSize  = (N + blockSize - 1) / blockSize;
    printf("launch config: grid = %d blocks, block = %d threads (total %d threads)\n",
           gridSize, blockSize, gridSize * blockSize);

    // Time the kernel using CUDA events (the only correct way).
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // Kernel launches are async. Sync the event before measuring.
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Kernel-internal errors surface here, not at the launch line.
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // --- Phase 6: D -> H copy ---
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- Phase 7: verify against CPU reference ---
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_C[i] - 3.0f) > 1e-5f) {
            if (errors < 5)
                printf("  mismatch at i=%d: got %f, expected 3.0\n", i, h_C[i]);
            ++errors;
        }
    }

    // --- Report ---
    // Kernel touches 3 arrays of N floats: read A, read B, write C = 3*bytes
    double gb_moved   = 3.0 * bytes / 1.0e9;
    double bw_gb_per_s = gb_moved / (ms / 1000.0);

    printf("\n--- results ---\n");
    printf("kernel time         : %.3f ms\n", ms);
    printf("effective bandwidth : %.1f GB/s\n", bw_gb_per_s);
    printf("peak bandwidth (A5000): 768.1 GB/s\n");
    printf("fraction of peak    : %.1f %%\n", 100.0 * bw_gb_per_s / 768.1);
    printf("correctness         : %s (%d errors)\n",
           errors == 0 ? "PASS" : "FAIL", errors);

    // --- Cleanup ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return errors == 0 ? 0 : 1;
}
