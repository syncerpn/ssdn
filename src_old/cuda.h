#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"
#define WARP_SIZE 32

void check_error(cudaError_t status);
void cuda_random(float *x_gpu, size_t n);
int *cuda_make_int_array(int *x, size_t n);
float cuda_compare(float *x_gpu, float *x, size_t n, char *s);
cublasHandle_t blas_handle();
cudnnHandle_t cudnn_handle();
dim3 cuda_gridsize(size_t n);
cudaStream_t get_cuda_stream();
int get_number_of_blocks(int array_size, int block_size);

#endif