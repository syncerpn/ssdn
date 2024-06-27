#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "curand_kernel.h"

#include "blas.h"
#include "cuda.h"
#include "utils.h"

#include <assert.h>

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) X[i*INCX] = ALPHA;
}

void fill_gpu(int N, float ALPHA, float *X, int INCX) {
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int INCX,  float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] += ALPHA * X[i*INCX];
}

void axpy_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] *= X[i*INCX];
}

void mul_gpu(int N, float *X, int INCX, float *Y, int INCY) {
    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void copy_kernel(int N,  float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] = X[i*INCX];
}

void copy_gpu(int N, float *X, int INCX, float *Y, int INCY) {
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void add_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] = X[i*INCX] + ALPHA;
}

void add_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void scale_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] = X[i*INCX] * ALPHA;
}

void scale_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    scale_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void min_kernel(int N, float MIN, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] = fmaxf(MIN, X[i*INCX]);
}

void min_gpu(int N, float MIN, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    min_kernel<<<cuda_gridsize(N), BLOCK>>>(N, MIN, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void max_kernel(int N, float MAX, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] = fminf(MAX, X[i*INCX]);
}

void max_gpu(int N, float MAX, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    max_kernel<<<cuda_gridsize(N), BLOCK>>>(N, MAX, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size) {
    int num = n*size*batch;
    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}