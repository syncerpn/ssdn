#ifndef _CUDA_H
#define _CUDA_H

#include "ssdn.h"
#define WARP_SIZE 32

void check_error(cudaError_t status);
dim3 cuda_gridsize(size_t n);
cudnnHandle_t cudnn_handle();
cublasHandle_t blas_handle();
void cuda_random(float *x_gpu, size_t n);
cudaStream_t get_cuda_stream();
int get_number_of_blocks(int array_size, int block_size);

#endif