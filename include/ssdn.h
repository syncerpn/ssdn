#ifndef SSDN_API
#define SSDN_API
#define BLOCK 512
extern int gpu_index;
extern unsigned int seed;

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"

void cuda_set_device(int n);

template <typename T>
T *cuda_make_array(T *x, size_t n) {
    T *x_gpu;
    size_t size = sizeof(T) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if (x) {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if (!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

template <typename T>
void cuda_free(T *x_gpu) {
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

template <typename T>
void cuda_push_array(T *x_gpu, T *x, size_t n) {
    size_t size = sizeof(T) * n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

template <typename T>
void cuda_pull_array(T *x_gpu, T *x, size_t n) {
    size_t size = sizeof(T) * n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

#endif