#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

extern "C" {
#include "blas.h"
#include "cuda.h"
#include "utils.h"
}

#include "curand_kernel.h"

__inline__ __device__
float warpAllReduceSum(float val) {
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
#if CUDART_VERSION >= 9000
        val += __shfl_xor_sync(0xffffffff, val, mask);
#else
        val += __shfl_xor(val, mask);
#endif
    return val;
}

__global__ void deconv_transpose_weights_kernels(float* weights, int spatial_size, int channels) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= channels) return;
    float swap;
    int j;
    for (j = 0; j < spatial_size/2; ++j) {
        swap = weights[i * spatial_size + j];
        weights[i * spatial_size + j] = weights[i * spatial_size + spatial_size - 1 - j];
        weights[i * spatial_size + spatial_size - 1 - j] = swap;
    }
}

void deconv_transpose_weights(float* weights, int spatial_size, int channels) {
    deconv_transpose_weights_kernels<<<cuda_gridsize(channels), BLOCK>>>(weights, spatial_size, channels);
    check_error(cudaPeekAtLastError());
}

__global__ void stretch_fill_3d_kernel(float* x, float* y, int w, int pad_w, int h, int pad_h, int c, int pad_c) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int ow = w + (w - 1) * pad_w;
    int oh = h + (h - 1) * pad_h;
    int oc = c + (c - 1) * pad_c;
    if (i >= ow * oh * oc) return;
    int iy = i;
    int iw = i % ow;
    i = (i - iw) / ow;
    int ih = i % oh;
    i = (i - ih) / oh;
    int ic = i;
    if (iw % (pad_w + 1) || ih % (pad_h + 1) || ic % (pad_c + 1)) {
        y[iy] = 0;
    } else {
        iw = iw / (pad_w + 1);
        ih = ih / (pad_h + 1);
        ic = ic / (pad_c + 1);
        y[iy] = x[ic * w * h + ih * w + iw];
    }
}

void stretch_fill_3d_gpu(float* x, float* y, int w, int pad_w, int h, int pad_h, int c, int pad_c) {
    //stretch x, save to y
    int ow = w + (w - 1) * pad_w;
    int oh = h + (h - 1) * pad_h;
    int oc = c + (c - 1) * pad_c;
    stretch_fill_3d_kernel<<<cuda_gridsize(ow * oh * oc), BLOCK>>>(x, y, w, pad_w, h, pad_h, c, pad_c);
    check_error(cudaPeekAtLastError());
}

__global__ void squeeze_fill_3d_kernel(float* x, float* y, int w, int pad_w, int h, int pad_h, int c, int pad_c) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= w * h * c) return;
    int iy = i;
    int iw = i % w;
    i = (i - iw) / w;
    int ih = i % h;
    i = (i - ih) / h;
    int ic = i;

    int ow = w + (w - 1) * pad_w;
    int oh = h + (h - 1) * pad_h;
    
    iw = iw * (pad_w + 1);
    ih = ih * (pad_h + 1);
    ic = ic * (pad_c + 1);
    y[iy] += x[ic * ow * oh + ih * ow + iw];
}

void squeeze_fill_3d_gpu(float* x, float* y, int w, int pad_w, int h, int pad_h, int c, int pad_c) {
    //squeeze x, save to y
    squeeze_fill_3d_kernel<<<cuda_gridsize(w * h * c), BLOCK>>>(x, y, w, pad_w, h, pad_h, c, pad_c);
    check_error(cudaPeekAtLastError());
}

__global__ void scale_bias_kernel(float *output, float *biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

void scale_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_scale_kernel(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    backward_scale_kernel<<<n, BLOCK>>>(x_norm, delta, batch, n, size, scale_updates);
    check_error(cudaPeekAtLastError());
}

__global__ void add_bias_kernel(float *output, float *biases, int batch, int n, int size)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

void add_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_bias_conn_kernel(float *bias_updates, float *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}

__global__ void backward_bias_kernel(float *bias_updates, float *delta, int batch, int n, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    if(size == 1){
        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>(bias_updates, delta, batch, n);
    }else{
        backward_bias_kernel<<<n, BLOCK>>>(bias_updates, delta, batch, n, size);
    }
    check_error(cudaPeekAtLastError());
}

__global__ void adam_kernel(int N, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;

    float mhat = m[index] / (1.f - powf(B1, t));
    float vhat = v[index] / (1.f - powf(B2, t));
    
    x[index] = x[index] + rate * mhat / (sqrtf(vhat) + eps);
}

extern "C" void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t)
{
    adam_kernel<<<cuda_gridsize(n), BLOCK>>>(n, x, m, v, B1, B2, rate, eps, t);
    check_error(cudaPeekAtLastError());
}

extern "C" void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_gpu(n, B1, m, 1);
    scal_gpu(n, B2, v, 1);
    axpy_gpu(n, -decay*batch, w, 1, d, 1);

    axpy_gpu(n, (1-B1), d, 1, m, 1);
    mul_gpu(n, d, 1, d, 1);
    axpy_gpu(n, (1-B2), d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
    fill_gpu(n, 0, d, 1);
}

__global__ void normalize_kernel(int N, float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    x[index] = (x[index] - mean[f])/(sqrtf(variance[f] + .00001f));
}

__global__ void normalize_delta_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    delta[index] = delta[index] * 1.f/(sqrtf(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}

extern "C" void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    size_t N = batch*filters*spatial;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, mean_delta, variance_delta, batch, filters, spatial, delta);
    check_error(cudaPeekAtLastError());
}

__global__ void variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5f * powf(variance[i] + .00001f, (float)(-3.f/2.f));
}

__global__ void accumulate_kernel(float *x, int n, int groups, float *sum)
{
    int k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}

__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}

__global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5f * powf(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}

__global__ void mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1.f/sqrtf(variance[i] + .00001f));
}

extern "C" void mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    fast_mean_delta_kernel<<<filters, BLOCK>>>(delta, variance, batch, filters, spatial, mean_delta);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    fast_variance_delta_kernel<<<filters, BLOCK>>>(x, delta, mean, variance, batch, filters, spatial, variance_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1.f/(batch * spatial);
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__global__ void variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    float scale = 1.f/(batch * spatial - 1);
    int j,k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += powf((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__global__ void reorg_kernel(int N, float ALPHA, float BETA, float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

    if(forward) out[out_index] = ALPHA * x[in_index] + BETA * out[out_index];
    else out[in_index] = ALPHA * x[out_index] + BETA * out[in_index];
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

__global__ void pow_kernel(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

__global__ void const_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void constrain_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = fminf(ALPHA, fmaxf(-ALPHA, X[i*INCX]));
}

__global__ void supp_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}

__global__ void add_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] += ALPHA;
}

__global__ void scal_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] *= ALPHA;
}

__global__ void floorf_kernel(int N, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = floorf(X[i*INCX]);
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void fill_int_kernel(int N, int ALPHA, int *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void copy_kernel(int N,  float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

__global__ void mul_kernel(int N, float *X, int INCX, float *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}

extern "C" void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

__global__ void l2norm_kernel(int N, float *x, float *dx, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += powf(x[index], 2);
    }
    sum = sqrtf(sum);
    if(sum == 0) sum = 1;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
        dx[index] = (1 - x[index]) / sum;
    }
}

extern "C" void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial)
{
    size_t N = batch*spatial;
    l2norm_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, dx, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

__global__ void fast_mean_kernel(float *x, int batch, int filters, int spatial, float *mean)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

__global__ void  fast_variance_kernel(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? powf((x[index] - mean[filter]), 2) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

extern "C" void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    fast_mean_kernel<<<filters, BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    fast_variance_kernel<<<filters, BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}


extern "C" void mean_gpu(float *x, int batch, int filters, int spatial, float *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}

extern "C" void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}

extern "C" void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

extern "C" void pow_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

extern "C" void mul_gpu(int N, float * X, int INCX, float * Y, int INCY)
{
    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX, Y, INCY);
    check_error(cudaPeekAtLastError());
}

extern "C" void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void flatten_kernel(int N, float ALPHA, float BETA, float *x, int spatial, int layers, int batch, int forward, float *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward) out[i2] = ALPHA * x[i1] + BETA * out[i2];
    else out[i1] = ALPHA * x[i2] + BETA * out[i1];
}

extern "C" void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out, float ALPHA, float BETA)
{
    int size = spatial*batch*layers;
    flatten_kernel<<<cuda_gridsize(size), BLOCK>>>(size, ALPHA, BETA, x, spatial, layers, batch, forward, out);
    check_error(cudaPeekAtLastError());
}

extern "C" void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out, float ALPHA, float BETA)
{
    int size = w*h*c*batch;
    reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, ALPHA, BETA, x, w, h, c, batch, stride, forward, out);
    check_error(cudaPeekAtLastError());
}

__global__ void mask_kernel(int n,  float *x, float mask_num, float *mask, float val)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] = val;
}

extern "C" void mask_gpu(int N, float * X, float mask_num, float * mask, float val)
{
    mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, val);
    check_error(cudaPeekAtLastError());
}

__global__ void scale_mask_kernel(int n,  float *x, float mask_num, float *mask, float scale)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}

extern "C" void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale)
{
    scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, mask_num, mask, scale);
    check_error(cudaPeekAtLastError());
}

extern "C" void const_gpu(int N, float ALPHA, float * X, int INCX)
{
    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void constrain_gpu(int N, float ALPHA, float * X, int INCX)
{
    constrain_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}


extern "C" void add_gpu(int N, float ALPHA, float * X, int INCX)
{
    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void scal_gpu(int N, float ALPHA, float * X, int INCX)
{
    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void floorf_gpu(int N, float * X, int INCX)
{
    floorf_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void supp_gpu(int N, float ALPHA, float * X, int INCX)
{
    supp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

extern "C" void fill_int_gpu(int N, int ALPHA, int * X, int INCX)
{
    fill_int_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = s1*out[out_index] + s2*add[add_index];
}

extern "C" void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
    check_error(cudaPeekAtLastError());
}

__global__ void smooth_l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        float abs_val = fabsf(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}

extern "C" void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    smooth_l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

__global__ void softmax_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}


extern "C" void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    softmax_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

__global__ void logistic_x_ent_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p+.0000001) - (1-t)*log(1-p+.0000001);
        delta[i] = t-p;
    }
}

extern "C" void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    logistic_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

__global__ void l2_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = diff * diff; //I know this is technically wrong, deal with it.
        delta[i] = 2 * diff;
    }
}

extern "C" void l2_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    l2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

__global__ void l1_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = abs(diff);
        delta[i] = (diff > 0) ? 1 : -1;
    }
}

extern "C" void l1_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

//nghiant_20190822:
//symmetric exp loss
__global__ void symexp_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = expf(diff) + expf(-diff) - 2;
        delta[i] = expf(diff) - expf(-diff);
    }
}

extern "C" void symexp_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    symexp_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

//logcosh loss
__global__ void logcosh_kernel(int n, float *pred, float* truth, float* delta, float* error) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = log(cosh(diff));
        delta[i] = expf(diff) - expf(-diff);
    }
}

extern "C" void logcosh_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    logcosh_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}
//nghiant_20190822_end

__global__ void wgan_kernel(int n, float *pred, float *truth, float *delta, float *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        error[i] = truth[i] ? -pred[i] : pred[i];
        delta[i] = (truth[i] > 0) ? 1 : -1;
    }
}

extern "C" void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error)
{
    wgan_kernel<<<cuda_gridsize(n), BLOCK>>>(n, pred, truth, delta, error);
    check_error(cudaPeekAtLastError());
}

__global__ void weighted_sum_kernel(int n, float *a, float *b, float *s, float *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

__global__ void deinter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            if(X) X[b*NX + j] += OUT[i];
        } else {
            if(Y) Y[b*NY + j - NX] += OUT[i];
        }
    }
}

extern "C" void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    deinter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
    check_error(cudaPeekAtLastError());
}

__global__ void inter_kernel(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            OUT[i] = X[b*NX + j];
        } else {
            OUT[i] = Y[b*NY + j - NX];
        }
    }
}

extern "C" void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    inter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, X, NY, Y, B, OUT);
    check_error(cudaPeekAtLastError());
}

extern "C" void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c)
{
    weighted_sum_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, c);
    check_error(cudaPeekAtLastError());
}

__global__ void weighted_delta_kernel(int n, float *a, float *b, float *s, float *da, float *db, float *ds, float *dc)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

extern "C" void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc)
{
    weighted_delta_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, s, da, db, ds, dc);
    check_error(cudaPeekAtLastError());
}

__global__ void mult_add_into_kernel(int n, float *a, float *b, float *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] += a[i]*b[i];
    }
}

extern "C" void mult_add_into_gpu(int num, float *a, float *b, float *c)
{
    mult_add_into_kernel<<<cuda_gridsize(num), BLOCK>>>(num, a, b, c);
    check_error(cudaPeekAtLastError());
}


__device__ void softmax_device(float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        float val = input[i*stride]; //nghiant_20191105: int??? should it be float instead?? -> changed to float
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = expf(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


__global__ void softmax_tree_kernel(float *input, int spatial, int batch, int stride, float temp, float *output, int groups, int *group_size, int *group_offset)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}

extern "C" void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier)
{
    int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
    int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
    
    int num = spatial*batch*hier.groups;
    softmax_tree_kernel<<<cuda_gridsize(num), BLOCK>>>(input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
    check_error(cudaPeekAtLastError());
    cuda_free((float *)tree_groups_size);
    cuda_free((float *)tree_groups_offset);
}

__global__ void softmax_kernel(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

extern "C" void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
    check_error(cudaPeekAtLastError());
}


__global__ void upsample_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if(forward) out[out_index] += scale * x[in_index];
    else atomicAdd(x+in_index, scale * out[out_index]);
}
extern "C" void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out);
    check_error(cudaPeekAtLastError());
}

__global__ void stack_kernel(size_t N, float *x, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%w;
    i = i/w;
    int out_h = i%h;
    i = i/h;
    int out_c = i%(c*stride);
    i = i/(c*stride);
    int b = i%batch;

    int in_w = out_w;
    int in_h = out_h;
    int in_c = out_c / stride;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;


    if(forward) out[out_index] += scale * x[in_index];
    else atomicAdd(x+in_index, scale * out[out_index]);
}

extern "C" void stack_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch*stride;
    stack_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out);
    check_error(cudaPeekAtLastError());
}

__global__ void sr_flat_kernel(size_t N, float *x, int w, int h, int c, int sr_scale, int batch, int forward, float scale, float *out)
{
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i % w;
    i = i / w;
    int in_h = i % h;
    i = i / h;
    int in_c = i % c;
    i = i / c;
    int b = i % batch;

    int u = in_c % (sr_scale * sr_scale);
    
    int out_w = in_w * sr_scale + u % sr_scale;
    int out_h = in_h * sr_scale + u / sr_scale;
    int out_c = in_c / (sr_scale * sr_scale);

    int out_index = b*w*h*c + out_c*w*sr_scale*h*sr_scale + out_h*w*sr_scale + out_w;


    if(forward) out[out_index] += scale * x[in_index];
    else x[in_index] += scale * out[out_index];
}

extern "C" void sr_flat_gpu(float *in, int w, int h, int c, int sr_scale, int batch, int forward, float scale, float *out)
{
    size_t size = w*h*c*batch;
    sr_flat_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, sr_scale, batch, forward, scale, out);
    check_error(cudaPeekAtLastError());
}

__global__ void gradient_clipping_kernel(float* x, float N, float clip, float norm) {
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    x[i] = clip * x[i] / norm;
}

extern "C" void gradient_clipping_gpu(float* x, float N, float clip, float norm) {
    gradient_clipping_kernel<<<cuda_gridsize(N), BLOCK>>>(x, N, clip, norm);
    check_error(cudaPeekAtLastError());
}

//nghiant: norm_w stuff
__global__ void norm_w_normalize_kernel(float* v, float* v_norm, int size, int n) {
    int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (f >= n) return;
    int i = 0;
    v_norm[f] = 0;
    for(i = 0; i < size; ++i){
        v_norm[f] += v[f*size + i] * v[f*size + i];
    }
    v_norm[f] = sqrtf(v_norm[f]);
}

extern "C" void norm_w_normalize_gpu(float* v, float* v_norm, int size, int n) {
    norm_w_normalize_kernel<<<cuda_gridsize(n), BLOCK>>>(v, v_norm, size, n);
    check_error(cudaPeekAtLastError());
}

__global__ void norm_w_scale_g_weights_kernel(float* weights, float* g, int size, int n) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n * size) return;
    int j = index % size;
    int i = (index - j) / size;
    weights[index] = weights[index] * g[i];
}

extern "C" void norm_w_scale_g_weights_gpu(float* weights, float* g, int size, int n) {
    norm_w_scale_g_weights_kernel<<<cuda_gridsize(n*size), BLOCK>>>(weights, g, size, n);
    check_error(cudaPeekAtLastError());
}

__global__ void norm_w_reform_weights_kernel(float* weights, float* v, float* v_norm, float* g, int size, int n) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n * size) return;
    int j = index % size;
    int i = (index - j) / size;
    weights[index] = v[index] / v_norm[i] * g[i];
}

extern "C" void norm_w_reform_weights_gpu(float* weights, float* v, float* v_norm, float* g, int size, int n) {
    norm_w_reform_weights_kernel<<<cuda_gridsize(n*size), BLOCK>>>(weights, v, v_norm, g, size, n);
    check_error(cudaPeekAtLastError());
}

__global__ void norm_w_norm_div_weights_kernel(float* weights, float* v, float* v_norm, int size, int n) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n * size) return;
    int j = index % size;
    int i = (index - j) / size;
    weights[index] = v[index] / v_norm[i];
}

extern "C" void norm_w_norm_div_weights_gpu(float* weights, float* v, float* v_norm, int size, int n) {
    norm_w_norm_div_weights_kernel<<<cuda_gridsize(n*size), BLOCK>>>(weights, v, v_norm, size, n);
    check_error(cudaPeekAtLastError());
}

__global__ void norm_w_delta_g_kernel(float* g_delta, float* weights_delta, float* v, float* v_norm, int size, int n) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j, index;
    g_delta[i] = 0;
    for (j = 0; j < size; ++j) {
        index = i * size + j;
        g_delta[i] += weights_delta[index] * v[index] / v_norm[i];
    }
}

extern "C" void norm_w_delta_g_gpu(float* g_delta, float* weights_delta, float* v, float* v_norm, int size, int n) {
    norm_w_delta_g_kernel<<<cuda_gridsize(n), BLOCK>>>(g_delta, weights_delta, v, v_norm, size, n);
    check_error(cudaPeekAtLastError());
}

__global__ void norm_w_delta_v_kernel(float* v_delta, float* weights_delta, float* g_delta, float* v, float* v_norm, float* g, int size, int n) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n * size) return;
    int j = index % size;
    int i = (index - j) / size;
    v_delta[index] = g[i] / v_norm[i] * weights_delta[index] - g[i] * g_delta[i] * v[index] / (v_norm[i] * v_norm[i]);
}

extern "C" void norm_w_delta_v_gpu(float* v_delta, float* weights_delta, float* g_delta, float* v, float* v_norm, float* g, int size, int n) {
    norm_w_delta_v_kernel<<<cuda_gridsize(n*size), BLOCK>>>(v_delta, weights_delta, g_delta, v, v_norm, g, size, n);
    check_error(cudaPeekAtLastError());
}
//nghiant_end

//nghiant: sampling stuff

__global__ void setup_curand_kernel(void* vstate, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    curandState_t* state = (curandState_t*)(vstate);
    curand_init(155061994, i, 0, &state[i]);
}

void* setup_curand_gpu(int n) {
    void* vstate;
    cudaMalloc((void**)(&vstate), n*sizeof(curandState_t));
    setup_curand_kernel<<<cuda_gridsize(n), BLOCK>>>(vstate, n);
    check_error(cudaPeekAtLastError());
    return vstate;
}
//nghiant_end

//nghiant: sam_layer, derived from darknet by alexeyab
__global__ void sam_kernel(float *in_w_h_c, int size, int channel_size, float *scales_c, float *out)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        out[index] = in_w_h_c[index] * scales_c[index];
    }
}

extern "C" void sam_gpu(float *in_w_h_c, int size, int channel_size, float *scales_c, float *out)
{
    const int num_blocks = get_number_of_blocks(size, BLOCK);

    sam_kernel<<<num_blocks, BLOCK, 0, get_cuda_stream()>>>(in_w_h_c, size, channel_size, scales_c, out);
    check_error(cudaPeekAtLastError());
}


__global__ void backward_sam_kernel(float *in_w_h_c_delta, int size, int channel_size,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        out_state_delta[index] += in_w_h_c_delta[index] * in_from_output[index]; // l.delta * from  (should be divided by channel_size?) ; //out_state_delta[index] += in_w_h_c_delta[index];
        out_from_delta[index] += in_scales_c[index] * in_w_h_c_delta[index]; // input * l.delta ; //out_from_delta[index] = in_w_h_c_delta[index];
    }
}

extern "C" void backward_sam_gpu(float *in_w_h_c_delta, int size, int channel_size,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int num_blocks = get_number_of_blocks(size, BLOCK);
    backward_sam_kernel<<<num_blocks, BLOCK, 0, get_cuda_stream()>>>(in_w_h_c_delta, size, channel_size, in_scales_c, out_from_delta, in_from_output, out_state_delta);
    check_error(cudaPeekAtLastError());
}

//nghiant:scale_channels derived from darknet alexeyab
__global__ void scale_channels_kernel(float *in_w_h_c, int size, int channel_size, int batch_size, int scale_wh, float *scales_c, float *out)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size) {
        if (scale_wh) {
            int osd_index = index % channel_size + (index / batch_size)*channel_size;

            out[index] = in_w_h_c[index] * scales_c[osd_index];
        }
        else {
            out[index] = in_w_h_c[index] * scales_c[index / channel_size];
        }
    }
}

extern "C" void scale_channels_gpu(float *in_w_h_c, int size, int channel_size, int batch_size, int scale_wh, float *scales_c, float *out)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    scale_channels_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> >(in_w_h_c, size, channel_size, batch_size, scale_wh, scales_c, out);
    check_error(cudaPeekAtLastError());
}




__global__ void backward_scale_channels_kernel(float *in_w_h_c_delta, int size, int channel_size, int batch_size, int scale_wh,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int index = blockIdx.x*blockDim.x + threadIdx.x;

    if (index < size) {

        if (scale_wh)
        {
            int osd_index = index % channel_size + (index / batch_size)*channel_size;

            //out_state_delta[osd_index] += in_w_h_c_delta[index] * in_from_output[index]; // l.delta * from  (should be divided by channel_size?)
            atomicAdd(&out_state_delta[osd_index], in_w_h_c_delta[index] * in_from_output[index] / channel_size); // l.delta * from

            out_from_delta[index] += in_scales_c[osd_index] * in_w_h_c_delta[index]; // input * l.delta  // atomic isn't required here

        }
        else {
            int osd_index = index / channel_size;
            //out_state_delta[osd_index] += in_w_h_c_delta[index] * in_from_output[index]; // l.delta * from  (should be divided by channel_size?)

            int warp_id = index / 32;
            int index_warp_start = warp_id * 32;
            int osd_index_warp_start = index_warp_start / channel_size;
            int osd_index_warp_end = (index_warp_start + 31) / channel_size;

            if (osd_index_warp_start == osd_index_warp_end) // all thread in warp process the same channel
            {
                float sum = warpAllReduceSum(in_w_h_c_delta[index] * in_from_output[index]); // l.delta * from
                if (threadIdx.x % 32 == 0) {
                    atomicAdd(&out_state_delta[osd_index], sum);
                    //out_state_delta[osd_index] += sum;
                }
            }
            else {
                atomicAdd(&out_state_delta[osd_index], in_w_h_c_delta[index] * in_from_output[index]); // l.delta * from
            }

            out_from_delta[index] += in_scales_c[osd_index] * in_w_h_c_delta[index]; // input * l.delta  // atomic isn't required here
        }
    }
}

extern "C" void backward_scale_channels_gpu(float *in_w_h_c_delta, int size, int channel_size, int batch_size, int scale_wh,
    float *in_scales_c, float *out_from_delta,
    float *in_from_output, float *out_state_delta)
{
    const int block_size = BLOCK;
    const int num_blocks = get_number_of_blocks(size, block_size);
    backward_scale_channels_kernel << <num_blocks, block_size, 0, get_cuda_stream() >> > (in_w_h_c_delta, size, channel_size, batch_size, scale_wh,
        in_scales_c, out_from_delta,
        in_from_output, out_state_delta);

    check_error(cudaPeekAtLastError());
}

//nghiant_end