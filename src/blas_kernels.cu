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

__global__ void axpy_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
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

__global__ void copy_kernel(int N, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[i*INCY] = X[i*INCX];
}

void copy_gpu(int N, float *X, int INCX, float *Y, int INCY) {
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void accumulate_kernel(int N, int K, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int j = 0; j < K; ++j) {
            Y[i*INCY] += X[(i*K+j)*INCX];
        }
    }
}

void accumulate_gpu(int N, int K, float *X, int INCX, float *Y, int INCY) {
    accumulate_kernel<<<cuda_gridsize(N), BLOCK>>>(N, K, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void tile_repeat_kernel(int N, int K, int M, float *X, int INCX, float *Y, int INCY) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= M*N) return;
    int k = index % K;
    index /= K;
    int j = index % M;
    index /= M;
    int i = index;
    Y[(i*K*M+j*K+k)*INCY] = X[(i*K+k)*INCX];
}

void tile_repeat_gpu(int N, int K, int M, float *X, int INCX, float *Y, int INCY) {
    tile_repeat_kernel<<<cuda_gridsize(N*M), BLOCK>>>(N, K, M, X, INCX, Y, INCY);
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

__global__ void padding_kernel(float* X, int w, int h, int c, int p, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int pw = w + 2 * p;
    int ph = h + 2 * p;
    if (index >= pw*ph*c) return;
    int wi = index % pw;
    index /= pw;
    int hi = index % ph;
    index /= ph;
    int ci = index;
    if (wi < p || wi >= pw - p || hi < p || hi >= ph - p) Y[ci*ph*pw+hi*pw+wi] = 0;
    else Y[ci*ph*pw+hi*pw+wi] = X[ci*h*w+(hi-p)*w+(wi-p)];
}

void padding_gpu(float* X, int w, int h, int c, int p, float* Y) {
    int pw = w + 2 * p;
    int ph = h + 2 * p;
    padding_kernel<<<cuda_gridsize(ph*pw*c), BLOCK>>>(X, w, h, c, p, Y);
    check_error(cudaPeekAtLastError());
}

__global__ void unrolling_kernel(float* X, int w, int h, int c, int k, int s, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int yw = (w - k) / s + 1;
    int yh = (h - k) / s + 1;
    if (index >= yw*yh*k*k*c) return;
    int kj = index % k;
    index /= k;
    int ki = index % k;
    index /= k;
    int ci = index % c;
    index /= c;
    int wi = index % yw;
    index /= yw;
    int hi = index;
    Y[(hi*yw+wi)*c*k*k+ci*k*k+ki*k+kj] = X[ci*h*w+(hi*s+ki)*w+wi*s+kj];
}

void unrolling_gpu(float* X, int w, int h, int c, int k, int s, float* Y) {
    int yw = (w - k) / s + 1;
    int yh = (h - k) / s + 1;

    unrolling_kernel<<<cuda_gridsize(yh*yw*k*k*c), BLOCK>>>(X, w, h, c, k, s, Y);
    check_error(cudaPeekAtLastError());
}

__global__ void flatten_arrange_kernel(float* X, float* Z, int w, int h, int s, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*s*s) return;
    int sj = index % s;
    index /= s;
    int si = index % s;
    index /= s;
    int wi = index % w;
    index /= w;
    int hi = index;
    Y[(hi*s+si)*w*s+wi*s+sj] = X[hi*w+wi] + Z[(si*s+sj)*h*w+hi*w+wi];
}

void flatten_arrange_gpu(float* X, float* Z, int w, int h, int s, float* Y) {
    flatten_arrange_kernel<<<cuda_gridsize(w*h*s*s), BLOCK>>>(X, Z, w, h, s, Y);
    check_error(cudaPeekAtLastError());
}

__global__ void distribute_mul_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = X[i*(k*k*c)+j] * Z[ni*(k*k*c)+j];
}

void distribute_mul_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    distribute_mul_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y);
    check_error(cudaPeekAtLastError());
}

__global__ void quantize_kernel(int N, float step, int nbit, bool sign, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float pos_end = sign ?  (float)(1 << (nbit - 1)) - 1 : (1 << nbit) - 1;
    float neg_end = sign ? -(float)(1 << (nbit - 1))     : 0;
    float raw_q = ceilf(X[i*INCX] / step);
    Y[i*INCY] = raw_q > pos_end ? pos_end : (raw_q < neg_end ? neg_end : raw_q);
}

void quantize_gpu(int N, float step, int nbit, bool sign, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    quantize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, step, nbit, sign, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__device__ void lead_one_encode(int x, int &k, int &p) {
    if (x == 0 || x == 1) {
        k = 0;
        p = 0;
    }
    k = 0;
    int xx = x;
    while (xx > 1) {
        xx = xx >> 1;
        k += 1;
    }
    p = x - (1 << k);
}

__device__ int approximate(int i, int j) {
    if ((i == 0) || (j == 0)) return 0;

    int sign_i = i > 0 ? 1 : -1;
    int sign_j = j > 0 ? 1 : -1;

    i = sign_i * i;
    j = sign_j * j;

    int ki, pi, kj, pj;
    lead_one_encode(i, ki, pi);
    lead_one_encode(j, kj, pj);

    int p1 = pi * pj;
    int p2 = ((1 << ki) - pi) * ((1 << kj) - pj);
    int e = p1 < p2 ? p1 : p2;
    return sign_i * sign_j * (i * j - e);

}

__global__ void distribute_approximate_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]));
}

void distribute_approximate_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    distribute_approximate_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y);
    check_error(cudaPeekAtLastError());
}

__global__ void compensate_wp_kernel(int N, float p, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int Xi = (int)(abs(X[i*INCX]));
    if (Xi & (Xi - 1) == 0) {
        Y[i*INCY] = X[i*INCX];
    }
    Y[i*INCY] = roundf(X[i*INCX] + p * X[i*INCX]);
}

void compensate_wp_gpu(int N, float p, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    compensate_wp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, p, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void quantize_compensate_wp_kernel(int N, float step, int nbit, bool sign, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float pos_end = sign ?  (float)(1 << (nbit - 1)) - 1 : (1 << nbit) - 1;
    float neg_end = sign ? -(float)(1 << (nbit - 1))     : 0;
    float raw_q = ceilf(X[i*INCX] / step);
    int Xi = (int)(abs(raw_q));
    if (Xi & (Xi - 1) != 0) {
        raw_q = roundf(raw_q + 0.048 * raw_q);
    }
    Y[i*INCY] = raw_q > pos_end ? pos_end : (raw_q < neg_end ? neg_end : raw_q);
}

void quantize_compensate_wp_gpu(int N, float step, int nbit, bool sign, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    quantize_compensate_wp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, step, nbit, sign, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}
