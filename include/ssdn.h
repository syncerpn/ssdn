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
void axpy_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
void copy_gpu(int N, float *X, int INCX, float *Y, int INCY);

void accumulate_gpu(int N, int K, float *X, int INCX, float *Y, int INCY);
void tile_repeat_gpu(int N, int K, int M, float *X, int INCX, float *Y, int INCY);

void add_gpu(int N, float ALPHA, float *X, int INCX=1, float *Y=0, int INCY=1);
void scale_gpu(int N, float ALPHA, float *X, int INCX=1, float *Y=0, int INCY=1);
void pow_gpu(int N, float ALPHA, float *X, int INCX=1, float *Y=0, int INCY=1);
void min_gpu(int N, float MIN, float *X, int INCX=1, float *Y=0, int INCY=1);
void max_gpu(int N, float MAX, float *X, int INCX=1, float *Y=0, int INCY=1);

void padding_gpu(float* X, int w, int h, int c, int p, float* Y);
void unrolling_gpu(float* X, int w, int h, int c, int k, int s, float* Y);
void flatten_arrange_gpu(float* X, float* Z, int w, int h, int s, float* Y);
void distribute_mul_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y);

void quantize_gpu(int N, float step, int nbit, bool sign, float *X, int INCX=1, float *Y=0, int INCY=1);
void distribute_approximate_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y);
void compensate_wp_gpu(int N, float p, float *X, int INCX=1, float *Y=0, int INCY=1);
void quantize_compensate_wp_gpu(int N, float step, int nbit, bool sign, float *X, int INCX=1, float *Y=0, int INCY=1);
void compensate_log_gpu(int N, float p, float m, float *X, int INCX=1, float *Y=0, int INCY=1);

// blas.cpp
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);

void accumulate_cpu(int N, int K, float *X, int INCX, float *Y, int INCY);
void tile_repeat_cpu(int N, int K, int M, float *X, int INCX, float *Y, int INCY);

void add_cpu(int N, float ALPHA, float *X, int INCX=1, float *Y=0, int INCY=1);
void scale_cpu(int N, float ALPHA, float *X, int INCX=1, float *Y=0, int INCY=1);
void pow_cpu(int N, float ALPHA, float *X, int INCX=1, float *Y=0, int INCY=1);
void min_cpu(int N, float MIN, float *X, int INCX=1, float *Y=0, int INCY=1);
void max_cpu(int N, float MAX, float *X, int INCX=1, float *Y=0, int INCY=1);

void padding_cpu(float* X, int w, int h, int c, int p, float* Y);
void unrolling_cpu(float* X, int w, int h, int c, int k, int s, float* Y);
void flatten_arrange_cpu(float* X, float* Z, int w, int h, int s, float* Y);
#endif