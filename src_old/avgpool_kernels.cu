#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "avgpool_layer.h"
#include "cuda.h"
}

__global__ void forward_avgpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, int dilation, float *input, float *output)
{
    int h = (in_h + pad - size - (size - 1) * (dilation - 1))/stride + 1;
    int w = (in_w + pad - size - (size - 1) * (dilation - 1))/stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));

    int l, m;
    int count = 0;
    float d = 0;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l*dilation;
            int cur_w = w_offset + j*stride + m*dilation;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            if (valid) {
                ++count;
                d += input[index];
            }
        }
    }
    output[out_index] = d / count;
}

__global__ void backward_avgpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, int dilation, float *delta, float *prev_delta, float impact)
{
    int h = (in_h + pad - size - (size - 1) * (dilation - 1))/stride + 1;
    int w = (in_w + pad - size - (size - 1) * (dilation - 1))/stride + 1;
    int c = in_c;
    int area = ((size-1)*dilation)/stride;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    float d = 0;
    int count = 0;
    int l, m;
    // is this a bug? original range is -area to area (< area+1) ???????????????????????????????
    for(l = -area; l <= 0; ++l){
        for(m = -area; m <= 0; ++m){
            int out_w = (j-w_offset)/stride + m;
            int out_h = (i-h_offset)/stride + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            if (valid) {
                d += delta[out_index];
                ++count;
            }
        }
    }
    prev_delta[index] += d / count * impact;
}

__global__ void forward_avgpool_layer_spatial_kernel(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__global__ void backward_avgpool_layer_spatial_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta, float impact)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] * impact / (w*h);
    }
}

extern "C" void forward_avgpool_layer_gpu(layer l, network net)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    size_t n = h*w*c*l.batch;

    if (l.spatial) {
        forward_avgpool_layer_spatial_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.w, l.h, l.c, net.input_gpu, l.output_gpu);
    } else {
        forward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.h, l.w, l.c, l.stride, l.size, l.pad, l.dilation, net.input_gpu, l.output_gpu);
    }
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_avgpool_layer_gpu(layer l, network net)
{
    size_t n = l.c*l.batch;

    if (l.spatial) {
        n = l.c * l.batch;
        backward_avgpool_layer_spatial_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.w, l.h, l.c, net.delta_gpu, l.delta_gpu, l.impact);
    } else {
        n = l.h*l.w*l.c*l.batch;
        backward_avgpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.h, l.w, l.c, l.stride, l.size, l.pad, l.dilation, l.delta_gpu, net.delta_gpu, l.impact);
    }
    check_error(cudaPeekAtLastError());
}