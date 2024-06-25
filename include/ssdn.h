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

// cuda.c
void cuda_set_device(int n);
float *cuda_make_array(float *x, size_t n);
void cuda_free(float *x_gpu);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);

// blas_kernels.cu
void fill_gpu(int N, float ALPHA, float *X, int INCX);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void axpy_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void add_gpu(int N, float ALPHA, float *X, int INCX);
void scale_gpu(int N, float ALPHA, float *X, int INCX);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
void copy_gpu(int N, float *X, int INCX, float *Y, int INCY);
void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void constrain_gpu(int N, float MIN, float MAX, float *X, int INCX);

// blas.cpp
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void add_cpu(int N, float ALPHA, float *X, int INCX);
void scale_cpu(int N, float ALPHA, float *X, int INCX);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void constrain_cpu(int N, float MIN, float MAX, float *X, int INCX);

#endif