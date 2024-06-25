#ifndef CUDA_H
#define CUDA_H

#include "ssdn.h"
#define WARP_SIZE 32

void check_error(cudaError_t status);
dim3 cuda_gridsize(size_t n);
cudnnHandle_t cudnn_handle();
cublasHandle_t blas_handle();
void cuda_random(float *x_gpu, size_t n);
cudaStream_t get_cuda_stream();
int get_number_of_blocks(int array_size, int block_size);

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