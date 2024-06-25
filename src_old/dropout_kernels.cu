#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void drop_out_kernel(float *input, float* output, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) output[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}

void forward_dropout_layer_gpu(layer l, network net)
{
    if (!net.train) {
        copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
        return;
    }
    int size = l.inputs*l.batch;
    cuda_random(l.rand_gpu, size);

    drop_out_kernel<<<cuda_gridsize(size), BLOCK>>>(net.input_gpu, l.output_gpu, size, l.rand_gpu, l.probability, l.scale);
    check_error(cudaPeekAtLastError());
}

__global__ void drop_out_kernel_backward(float *input, float* output, int size, float *rand, float prob, float scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) output[id] += (rand[id] < prob) ? 0 : input[id]*scale;
}

void backward_dropout_layer_gpu(layer l, network net)
{
    if(!net.delta_gpu) return;
    int size = l.inputs*l.batch;

    drop_out_kernel_backward<<<cuda_gridsize(size), BLOCK>>>(l.delta_gpu, net.delta_gpu, size, l.rand_gpu, l.probability, l.scale * l.impact);
    check_error(cudaPeekAtLastError());
}