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

__global__ void broadcast_mul_kernel(int N, int s, float *X, float *Z, float *Y) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N * s) Y[i] = X[i] * Z[i / s];
}

void broadcast_mul_gpu(int N, int s, float *X, float *Z, float *Y) {
    broadcast_mul_kernel<<<cuda_gridsize(N * s), BLOCK>>>(N, s, X, Z, Y);
    check_error(cudaPeekAtLastError());
}

__global__ void broadcast_add_kernel(int N, int s, float *X, float *Z, float *Y) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N * s) Y[i] = X[i] + Z[i / s];
}

void broadcast_add_gpu(int N, int s, float *X, float *Z, float *Y) {
    broadcast_add_kernel<<<cuda_gridsize(N * s), BLOCK>>>(N, s, X, Z, Y);
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

__global__ void pixel_shuffle_kernel(float* X, int w, int h, int c, int s, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*s*s*c) return;
    int sj = index % s;
    index /= s;
    int si = index % s;
    index /= s;
    int wi = index % w;
    index /= w;
    int hi = index % h;
    index /= h;
    int ci = index;
    Y[ci*s*s*h*w+(hi*s+si)*w*s+wi*s+sj] = X[ci*h*w*s*s+(si*s+sj)*h*w+hi*w+wi];
}

void pixel_shuffle_gpu(float* X, int w, int h, int c, int s, float* Y) {
    pixel_shuffle_kernel<<<cuda_gridsize(w*h*s*s*c), BLOCK>>>(X, w, h, c, s, Y);
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
    float raw_q = roundf(X[i*INCX] / step);
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

// 241121: now is the original mitchell's algorithm
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

// this is the key difference, where we no longer can make sure the error is min of the two possible errors
__device__ int approximate_one_side(int i, int j) {
    if ((i == 0) || (j == 0)) return 0;

    int sign_i = i > 0 ? 1 : -1;
    int sign_j = j > 0 ? 1 : -1;

    i = sign_i * i;
    j = sign_j * j;

    int ki, pi, kj, pj;
    lead_one_encode(i, ki, pi);
    lead_one_encode(j, kj, pj);

    int e = pi * pj;
    return sign_i * sign_j * (i * j - e);
}

__global__ void distribute_approximate_one_side_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate_one_side((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]));
}

void distribute_approximate_one_side_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    distribute_approximate_one_side_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y);
    check_error(cudaPeekAtLastError());
}

__device__ int approximate_one_side_iterative(int i, int j, int iter) {
    if ((i == 0) || (j == 0)) return 0;

    int sign_i = i > 0 ? 1 : -1;
    int sign_j = j > 0 ? 1 : -1;

    i = sign_i * i;
    j = sign_j * j;

    int ii = i;
    int jj = j;

    int ki, pi, kj, pj;

    while (iter > 0) {
        lead_one_encode(ii, ki, pi);
        lead_one_encode(jj, kj, pj);

        ii = pi;
        jj = pj;

        iter -= 1;
    }

    int e = pi * pj;
    return sign_i * sign_j * (i * j - e);
}

__global__ void distribute_approximate_one_side_iterative_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate_one_side_iterative((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]), iter);
}

void distribute_approximate_one_side_iterative_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    distribute_approximate_one_side_iterative_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y, iter);
    check_error(cudaPeekAtLastError());
}
//241121_end

// 241121: this is actually a customized version of mitchell's original algorithm
// with also a modified circuit
// and it works so well
// we will reuse the lead_one_encode function as well
// __device__ void lead_one_encode(int x, int &k, int &p) {}

__device__ int approximate_signedap(int i, int j) {
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
    if (sign_i * sign_j == -1) return -(i * j - e) - 1;
    return sign_i * sign_j * (i * j - e);
}

__global__ void distribute_approximate_signedap_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate_signedap((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]));
}

void distribute_approximate_signedap_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    distribute_approximate_signedap_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y);
    check_error(cudaPeekAtLastError());
}

__device__ int approximate_signedap_iterative(int i, int j, int iter) {
    if ((i == 0) || (j == 0)) return 0;

    int sign_i = i > 0 ? 1 : -1;
    int sign_j = j > 0 ? 1 : -1;

    i = sign_i * i;
    j = sign_j * j;

    int ii = i;
    int jj = j;

    int ki, pi, kj, pj;

    while (iter > 0) {
        lead_one_encode(ii, ki, pi);
        lead_one_encode(jj, kj, pj);

        ii = pi;
        jj = pj;

        iter -= 1;
    }

    int p1 = pi * pj;
    int p2 = ((1 << ki) - pi) * ((1 << kj) - pj);
    int e = p1 < p2 ? p1 : p2;
    if ((sign_i * sign_j) == -1) return -(i * j - e) - 1;
    return sign_i * sign_j * (i * j - e);
}

__global__ void distribute_approximate_signedap_iterative_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate_signedap_iterative((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]), iter);
}

void distribute_approximate_signedap_iterative_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    distribute_approximate_signedap_iterative_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y, iter);
    check_error(cudaPeekAtLastError());
}
//241121_end

// 241121: this is actually a customized version of mitchell's original algorithm
// with also a modified circuit
// and it works so well
// we will reuse the lead_one_encode function as well
// __device__ void lead_one_encode(int x, int &k, int &p) {}

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

__device__ int approximate_iterative(int i, int j, int iter) {
    if ((i == 0) || (j == 0)) return 0;

    int sign_i = i > 0 ? 1 : -1;
    int sign_j = j > 0 ? 1 : -1;

    i = sign_i * i;
    j = sign_j * j;

    int ii = i;
    int jj = j;

    int ki, pi, kj, pj;

    while (iter > 0) {
        lead_one_encode(ii, ki, pi);
        lead_one_encode(jj, kj, pj);

        ii = pi;
        jj = pj;

        iter -= 1;
    }

    int p1 = pi * pj;
    int p2 = ((1 << ki) - pi) * ((1 << kj) - pj);
    int e = p1 < p2 ? p1 : p2;
    return sign_i * sign_j * (i * j - e);
}

__global__ void distribute_approximate_iterative_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate_iterative((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]), iter);
}

void distribute_approximate_iterative_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    distribute_approximate_iterative_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y, iter);
    check_error(cudaPeekAtLastError());
}
//241121_end

// 241102
// improved log multiplier approximation style
// implement this to simulate accuracy/psnr of networks
__device__ void nearest_one_encode(int x, int &k, int &p) {
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
    int p_over = (1 << (k+1)) - x;
    if (p_over < p) {
        p = -p_over; // must be negative
        k += 1;
    }
}

__device__ int approximate_ilm(int i, int j) {
    if ((i == 0) || (j == 0)) return 0;

    int sign_i = i > 0 ? 1 : -1;
    int sign_j = j > 0 ? 1 : -1;

    i = sign_i * i;
    j = sign_j * j;

    int ki, pi, kj, pj;
    nearest_one_encode(i, ki, pi);
    nearest_one_encode(j, kj, pj);

    int m = (1 << (ki + kj)) + (pi << kj) + (pj << ki);
    return sign_i * sign_j * m;
}

__global__ void distribute_approximate_ilm_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate_ilm((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]));
}

void distribute_approximate_ilm_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y) {
    distribute_approximate_ilm_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y);
    check_error(cudaPeekAtLastError());
}

__device__ int approximate_ilm_iterative(int i, int j, int iter) {
    int m = 0;
    int ii = i;
    int jj = j;

    while (iter > 0) {
        if ((ii == 0) || (jj == 0)) break;

        int sign_i = ii > 0 ? 1 : -1;
        int sign_j = jj > 0 ? 1 : -1;

        ii = sign_i * ii;
        jj = sign_j * jj;

        int ki, kj, pi, pj;

        nearest_one_encode(ii, ki, pi);
        nearest_one_encode(jj, kj, pj);

        m += sign_i * sign_j * ((1 << (ki + kj)) + (pi << kj) + (pj << ki));

        ii = sign_i * pi;
        jj = sign_j * pj;

        iter -= 1;
    }
    return m;
}

__global__ void distribute_approximate_ilm_iterative_kernel(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= w*h*c*k*k*n) return;
    int  j = index % (c*k*k);
    index /= (c*k*k);
    int  i = index % (h*w);
    index /= (h*w);
    int ni = index;

    Y[ni*(h*w*k*k*c)+i*(k*k*c)+j] = (float)approximate_ilm_iterative((int)(X[i*(k*k*c)+j]), (int)(Z[ni*(k*k*c)+j]), iter);
}

void distribute_approximate_ilm_iterative_gpu(float* X, float* Z, int w, int h, int c, int k, int n, float* Y, int iter) {
    distribute_approximate_ilm_iterative_kernel<<<cuda_gridsize(w*h*c*k*k*n), BLOCK>>>(X, Z, w, h, c, k, n, Y, iter);
    check_error(cudaPeekAtLastError());
}
// 241102_end

__global__ void compensate_wp_kernel(int N, float p, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int Xi = (int)(abs(X[i*INCX]));
    if ((Xi & (Xi - 1)) == 0) {
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

__global__ void quantize_compensate_wp_kernel(int N, float step, int nbit, float comp_factor, bool sign, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sign_m = X[i*INCX] > 0 ? 1.0 : -1.0;
    float pos_end = sign ?  (float)(1 << (nbit - 1)) - 1 : (1 << nbit) - 1;
    float neg_end = sign ? -(float)(1 << (nbit - 1))     : 0;
    float raw_q = roundf(abs(X[i*INCX]) / step);
    int Xi = (int)(raw_q);
    if ((Xi & (Xi - 1)) != 0) {
        raw_q += roundf(comp_factor * raw_q);
    }
    raw_q = raw_q * sign_m;
    Y[i*INCY] = raw_q > pos_end ? pos_end : (raw_q < neg_end ? neg_end : raw_q);
}

void quantize_compensate_wp_gpu(int N, float step, int nbit, float comp_factor, bool sign, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    quantize_compensate_wp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, step, nbit, comp_factor, sign, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void compensate_log_kernel(int N, float p, float m, float *X, int INCX, float *Y, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sign_m = X[i*INCX] > 0 ? 1.0 : -1.0;
    int Xi = (int)(abs(X[i*INCX]));
    if ((Xi & (Xi - 1)) == 0) {
        Y[i*INCY] = X[i*INCX];
    } else {
        float Xa = abs(X[i*INCX]);
        Y[i*INCY] = sign_m * roundf(Xa + p / m * (m - Xa) * Xa);
    }
}

void compensate_log_gpu(int N, float p, float m, float *X, int INCX, float *Y, int INCY) {
    if (Y == 0) {
        Y = X;
        INCY = INCX;
    }
    compensate_log_kernel<<<cuda_gridsize(N), BLOCK>>>(N, p, m, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

// specifically for non-dimensional change conv2d (3,1,1) or (1,1,0) for example
__global__ void conv2d_chunk_approximate_ilm_iterative_kernel(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= h*w*c) return;
    int ci = index % c;
    index /= c;
    int wi = index % w;
    index /= w;
    int hi = index;

    int ki, kj, xhi, xwi;

    Y[hi*w*c + wi*c + ci] = 0;

    for (ki = 0; ki < k; ++ki) {
        for (kj = 0; kj < k; ++kj) {
            xhi = hi + ki - k / 2;
            xwi = wi + kj - k / 2;
            if (0 <= xhi && xhi < h && 0 <= xwi && xwi < w) {
                Y[hi*w*c + wi*c + ci] += (float)approximate_ilm_iterative((int)(X[ci*h*w+xhi*w+xwi]), (int)(W[ci*k*k+ki*k+kj]), iter);
            }
        }
    }
}

void conv2d_chunk_approximate_ilm_iterative_gpu(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    conv2d_chunk_approximate_ilm_iterative_kernel<<<cuda_gridsize(h*w*c), BLOCK>>>(X, W, w, h, c, k, Y, iter);
    check_error(cudaPeekAtLastError());
}

// specifically for non-dimensional change conv2d (3,1,1) or (1,1,0) for example
__global__ void conv2d_chunk_approximate_iterative_kernel(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= h*w*c) return;
    int ci = index % c;
    index /= c;
    int wi = index % w;
    index /= w;
    int hi = index;

    int ki, kj, xhi, xwi;

    Y[hi*w*c + wi*c + ci] = 0;

    for (ki = 0; ki < k; ++ki) {
        for (kj = 0; kj < k; ++kj) {
            xhi = hi + ki - k / 2;
            xwi = wi + kj - k / 2;
            if (0 <= xhi && xhi < h && 0 <= xwi && xwi < w) {
                Y[hi*w*c + wi*c + ci] += (float)approximate_iterative((int)(X[ci*h*w+xhi*w+xwi]), (int)(W[ci*k*k+ki*k+kj]), iter);
            }
        }
    }
}

void conv2d_chunk_approximate_iterative_gpu(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    conv2d_chunk_approximate_iterative_kernel<<<cuda_gridsize(h*w*c), BLOCK>>>(X, W, w, h, c, k, Y, iter);
    check_error(cudaPeekAtLastError());
}

// specifically for non-dimensional change conv2d (3,1,1) or (1,1,0) for example
__global__ void conv2d_chunk_approximate_signedap_iterative_kernel(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= h*w*c) return;
    int ci = index % c;
    index /= c;
    int wi = index % w;
    index /= w;
    int hi = index;

    int ki, kj, xhi, xwi;

    Y[hi*w*c + wi*c + ci] = 0;

    for (ki = 0; ki < k; ++ki) {
        for (kj = 0; kj < k; ++kj) {
            xhi = hi + ki - k / 2;
            xwi = wi + kj - k / 2;
            if (0 <= xhi && xhi < h && 0 <= xwi && xwi < w) {
                Y[hi*w*c + wi*c + ci] += (float)approximate_signedap_iterative((int)(X[ci*h*w+xhi*w+xwi]), (int)(W[ci*k*k+ki*k+kj]), iter);
            }
        }
    }
}

void conv2d_chunk_approximate_signedap_iterative_gpu(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    conv2d_chunk_approximate_signedap_iterative_kernel<<<cuda_gridsize(h*w*c), BLOCK>>>(X, W, w, h, c, k, Y, iter);
    check_error(cudaPeekAtLastError());
}

// specifically for non-dimensional change conv2d (3,1,1) or (1,1,0) for example
__global__ void conv2d_chunk_approximate_one_side_iterative_kernel(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= h*w*c) return;
    int ci = index % c;
    index /= c;
    int wi = index % w;
    index /= w;
    int hi = index;

    int ki, kj, xhi, xwi;

    Y[hi*w*c + wi*c + ci] = 0;

    for (ki = 0; ki < k; ++ki) {
        for (kj = 0; kj < k; ++kj) {
            xhi = hi + ki - k / 2;
            xwi = wi + kj - k / 2;
            if (0 <= xhi && xhi < h && 0 <= xwi && xwi < w) {
                Y[hi*w*c + wi*c + ci] += (float)approximate_one_side_iterative((int)(X[ci*h*w+xhi*w+xwi]), (int)(W[ci*k*k+ki*k+kj]), iter);
            }
        }
    }
}

void conv2d_chunk_approximate_one_side_iterative_gpu(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    conv2d_chunk_approximate_one_side_iterative_kernel<<<cuda_gridsize(h*w*c), BLOCK>>>(X, W, w, h, c, k, Y, iter);
    check_error(cudaPeekAtLastError());
}

// specifically for non-dimensional change conv2d (3,1,1) or (1,1,0) for example
__global__ void conv2d_chunk_mul_kernel(float* X, float* W, int w, int h, int c, int k, float* Y) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= h*w*c) return;
    int ci = index % c;
    index /= c;
    int wi = index % w;
    index /= w;
    int hi = index;

    int ki, kj, xhi, xwi;

    Y[hi*w*c + wi*c + ci] = 0;

    for (ki = 0; ki < k; ++ki) {
        for (kj = 0; kj < k; ++kj) {
            xhi = hi + ki - k / 2;
            xwi = wi + kj - k / 2;
            if (0 <= xhi && xhi < h && 0 <= xwi && xwi < w) {
                Y[hi*w*c + wi*c + ci] += X[ci*h*w+xhi*w+xwi] * W[ci*k*k+ki*k+kj];
            }
        }
    }
}

void conv2d_chunk_mul_gpu(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    conv2d_chunk_mul_kernel<<<cuda_gridsize(h*w*c), BLOCK>>>(X, W, w, h, c, k, Y);
    check_error(cudaPeekAtLastError());
}

__device__ int approximate_lsbskip_iterative(int i, int j, int iter) {
    if ((i == 0) || (j == 0)) return 0;

    int m = 0;
    int sm = 1;
    int l = 0;
    int sl = 1;
    bool found = false;

    if ((-2 <= j) && (j <= 2)) {
        sl = j > 0 ? 1 : -1;
        l = j * sl;
    } else {
        for (sm = 1; sm > -2; sm -= 2) {
            for (m = 4; m < 2048; m *= 2) {
                for (sl = 1; sl > -2; sl -= 2) {
                    for (l = 0; l < 3; ++l) {
                        if (j == (sm * m + sl * l)) {
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                if (found) break;
            }
            if (found) break;
        }
    }

    int isl = i * l * sl;
    int ism = i * m * sm;

    return (((ism >> iter) + (isl >> iter)) << iter) | (isl & ((1 << iter) - 1));

}

// specifically for non-dimensional change conv2d (3,1,1) or (1,1,0) for example
__global__ void conv2d_chunk_approximate_lsbskip_iterative_kernel(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= h*w*c) return;
    int ci = index % c;
    index /= c;
    int wi = index % w;
    index /= w;
    int hi = index;

    int ki, kj, xhi, xwi;

    Y[hi*w*c + wi*c + ci] = 0;

    for (ki = 0; ki < k; ++ki) {
        for (kj = 0; kj < k; ++kj) {
            xhi = hi + ki - k / 2;
            xwi = wi + kj - k / 2;
            if (0 <= xhi && xhi < h && 0 <= xwi && xwi < w) {
                Y[hi*w*c + wi*c + ci] += (float)approximate_lsbskip_iterative((int)(X[ci*h*w+xhi*w+xwi]), (int)(W[ci*k*k+ki*k+kj]), iter);
            }
        }
    }
}

void conv2d_chunk_approximate_lsbskip_iterative_gpu(float* X, float* W, int w, int h, int c, int k, float* Y, int iter) {
    conv2d_chunk_approximate_lsbskip_iterative_kernel<<<cuda_gridsize(h*w*c), BLOCK>>>(X, W, w, h, c, k, Y, iter);
    check_error(cudaPeekAtLastError());
}