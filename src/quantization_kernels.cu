#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "quantization_layer.h"
#include "cuda.h"
}

// UNIFORM
__device__ float uniform_quantize_forward_kernel(float x, float step_size, int num_level) {
    float pos_end, neg_end, qx;
    float zero_center = num_level % 2;

    pos_end = step_size * ((num_level - 1) * 0.5);
    neg_end = -pos_end;

    qx = (round(x/step_size + (1 - zero_center) * 0.5) - 0.5 + zero_center * 0.5) * step_size;
    qx = qx > pos_end ? pos_end : ((qx < neg_end) ? neg_end : qx);

    return qx;
}

__device__ float uniform_quantize_backward_kernel(float x, float step_size, int num_level) {
    float pos_end, neg_end, qx;
    float zero_center = num_level % 2;

    pos_end = step_size * ((num_level - 1) * 0.5);
    neg_end = -pos_end;

    qx = (round(x/step_size + (1 - zero_center) * 0.5) - 0.5 + zero_center * 0.5) * step_size;

    if ((qx < neg_end) || (qx > pos_end)) return 0;
    else return 1;
}

// ROOT
__device__ float root_quantize_forward_kernel(float x, float step_size, int num_level, float root) {
    float bound, qx, max, min;

    bound = root + step_size * (num_level - 1);
    if (bound > root) {
        max = bound;
        min = root;
    } else {
        max = root;
        min = bound;
    }

    qx = round((x - root) / step_size) * step_size + root;
    qx = qx > max ? max : ((qx < min) ? min : qx);

    return qx;
}

__device__ float root_quantize_backward_kernel(float x, float step_size, int num_level, float root) {
    float bound, qx, max, min;

    bound = root + step_size * (num_level - 1);
    if (bound > root) {
        max = bound;
        min = root;
    } else {
        max = root;
        min = bound;
    }

    qx = round((x - root) / step_size) * step_size + root;

    if ((qx < min) || (qx > max)) return 0;
    else return 1;
}

// BIT (same as UNIFORM quantization but defined by bit-width)
__device__ float bit_quantize_forward_kernel(float x, float step_size, int bitwidth, int zero_center) {
    float pos_end, neg_end, qx;
    unsigned int num_level = 1 << bitwidth;

    pos_end = step_size * ((num_level - 1 - zero_center) * 0.5);
    neg_end = -pos_end - step_size * zero_center;

    qx = (round(x/step_size + (1 - zero_center) * 0.5) - 0.5 + zero_center * 0.5) * step_size;
    qx = qx > pos_end ? pos_end : ((qx < neg_end) ? neg_end : qx);

    return qx;
}

__device__ float bit_quantize_backward_kernel(float x, float step_size, int bitwidth, int zero_center) {
    float pos_end, neg_end, qx;
    unsigned int num_level = 1 << bitwidth;

    pos_end = step_size * ((num_level - 1 - zero_center) * 0.5);
    neg_end = -pos_end - step_size * zero_center;

    qx = (round(x/step_size + (1 - zero_center) * 0.5) - 0.5 + zero_center * 0.5) * step_size;

    if ((qx < neg_end) || (qx > pos_end)) return 0;
    else return 1;
}

// main kernel
__device__ float quantize_forward_kernel(float x, quantization_scheme qs) {
    switch (qs.type) {
    	case QS_UNIFORM:
    		return uniform_quantize_forward_kernel(x, qs.step_size, qs.num_level);
        case QS_ROOT:
            return root_quantize_forward_kernel(x, qs.step_size, qs.num_level, qs.root);
        case QS_BIT:
            return bit_quantize_forward_kernel(x, qs.step_size, qs.num_level, qs.zero_center);
    }
    return x;
}

__device__ float quantize_backward_kernel(float x, quantization_scheme qs) {
    switch (qs.type) {
    	case QS_UNIFORM:
    		return uniform_quantize_backward_kernel(x, qs.step_size, qs.num_level);
        case QS_ROOT:
            return root_quantize_backward_kernel(x, qs.step_size, qs.num_level, qs.root);
        case QS_BIT:
            return bit_quantize_backward_kernel(x, qs.step_size, qs.num_level, qs.zero_center);
    }
    return 1;
}

__global__ void quantize_array_forward_kernel(float* x, int n, quantization_scheme qs) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = quantize_forward_kernel(x[i], qs);
}

__global__ void quantize_array_backward_kernel(float* x, int n, quantization_scheme qs, float* delta) {
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) delta[i] *= quantize_backward_kernel(x[i], qs);
}

extern "C" void quantize_array_forward_gpu(float* x, int n, quantization_scheme qs) {
    quantize_array_forward_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, qs);
    check_error(cudaPeekAtLastError());
}

extern "C" void quantize_array_backward_gpu(float* x, int n, quantization_scheme qs, float* delta) {
    quantize_array_backward_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, qs, delta);
    check_error(cudaPeekAtLastError());
}

void forward_quantization_layer_gpu(layer l, network net) {
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    quantize_array_forward_gpu(l.output_gpu, l.outputs*l.batch, l.quantization);
}

void backward_quantization_layer_gpu(layer l, network net) {
    quantize_array_backward_gpu(l.output_gpu, l.outputs*l.batch, l.quantization, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}