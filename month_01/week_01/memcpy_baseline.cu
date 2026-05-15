// memcpy_baseline.cu - Day 3
// Device-to-device cudaMemcpy: ceiling for 2N-byte traffic kernels.
// Compare against transpose_* variants. A5000 spec peak: 768 GB/s.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    size_t bytes = (size_t)N * N * sizeof(float);
    printf("memcpy_baseline: N=%d, buffer=%.2f MB, traffic/copy=%.2f MB (2N)\n",
           N, bytes / 1e6, 2.0 * bytes / 1e6);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 0x42, bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int WARMUP = 5, TIMED = 20;
    for (int i = 0; i < WARMUP; i++)
        CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    float min_ms = 1e9f, sum_ms = 0.0f;
    for (int i = 0; i < TIMED; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms < min_ms) min_ms = ms;
        sum_ms += ms;
    }
    float avg_ms = sum_ms / TIMED;

    double traffic = 2.0 * (double)bytes;
    double min_bw = traffic / (min_ms * 1e-3) / 1e9;
    double avg_bw = traffic / (avg_ms * 1e-3) / 1e9;
    const double PEAK = 768.0;

    printf("\nResults (%d timed, %d warmup):\n", TIMED, WARMUP);
    printf("  Min: %.3f ms  ->  %.1f GB/s  (%.1f%% of peak)\n", min_ms, min_bw, 100.0 * min_bw / PEAK);
    printf("  Avg: %.3f ms  ->  %.1f GB/s  (%.1f%% of peak)\n", avg_ms, avg_bw, 100.0 * avg_bw / PEAK);
    printf("  A5000 spec peak: %.0f GB/s\n", PEAK);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
