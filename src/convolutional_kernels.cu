#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "quantization_layer.h"
}

void transform_weights(float *weights, int n, int size, float *tran_weights, weight_transform_scheme wts, float* q_coeff, int n_coeff)
{
    switch(wts.type){
        case WTS_MAX_SHIFTER:
            make_shifting_weights_max_gpu(weights, n, size, tran_weights, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_MEAN_SHIFTER:
            make_shifting_weights_mean_gpu(weights, n, size, tran_weights, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_SHIFTER:
            make_shifting_weights_gpu(weights, n * size, tran_weights, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_UNIFORM:
            uniform_quantize_weights_gpu(weights, n * size, tran_weights, wts.step_size, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_SCALE_LINEAR:
            make_scale_linear_gpu(weights, n, size, tran_weights, n_coeff);
            break;
        case WTS_NONE:
            break;
    }
}

void forward_convolutional_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    if (!net.pre_transform) {
        transform_weights(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.tran_weights_gpu, l.weight_transform, l.q_coeff_gpu, l.n_coeff);
    }
    
    if (l.weight_transform.type) swap_weight_transform(&l);

    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

    if (l.batch_normalize) forward_batchnorm_layer_gpu(l, net);
    else add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);

    if (l.quantization.type) quantize_array_forward_gpu(l.output_gpu, l.outputs*l.batch, l.quantization);

    if (l.weight_transform.type) swap_weight_transform(&l);
}

__global__ void smooth_kernel(float *x, int n, int w, int h, int c, int size, float rate, float *delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -(size/2.f);
    int h_offset = -(size/2.f);

    int out_index = j + w*(i + h*(k + c*b));
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i + l;
            int cur_w = w_offset + j + m;
            int index = cur_w + w*(cur_h + h*(k + b*c));
            int valid = (cur_h >= 0 && cur_h < h &&
                    cur_w >= 0 && cur_w < w);
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;
        }
    }
}

extern "C" void smooth_layer(layer l, int size, float rate)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;

    size_t n = h*w*c*l.batch;

    smooth_kernel<<<cuda_gridsize(n), BLOCK>>>(l.output_gpu, n, l.w, l.h, l.c, size, rate, l.delta_gpu);
    check_error(cudaPeekAtLastError());
}

void backward_convolutional_layer_gpu(layer l, network net)
{
    if(l.smooth) smooth_layer(l, 5, l.smooth);

    if(l.quantization.type) quantize_array_backward_gpu(l.output_gpu, l.outputs*l.batch, l.quantization, l.delta_gpu);

    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    if(l.batch_normalize) backward_batchnorm_layer_gpu(l, net);
    else backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

    if (l.clip) {
        cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.nbiases);
        int ii;
        float norm = 0;
        for (ii = 0; ii < l.nbiases; ++ii) {
            norm += l.bias_updates[ii] * l.bias_updates[ii];
        }
        norm = sqrt(norm);
        if (norm > l.clip * net.batch) {
            // fprintf(stderr, "[ INFO ] norm: %.6f > %.6f -> clipping\n", norm, l.clip * net.batch);
            gradient_clipping_gpu(l.bias_updates_gpu, l.nbiases, l.clip * net.batch, norm);
        }
    }
    
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if (l.clip) {
        cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
        int ii;
        float norm = 0;
        for (ii = 0; ii < l.nweights; ++ii) {
            norm += l.weight_updates[ii] * l.weight_updates[ii];
        }
        norm = sqrt(norm);
        if (norm > l.clip * net.batch) {
            // fprintf(stderr, "[ INFO ] norm: %.6f > %.6f -> clipping\n", norm, l.clip * net.batch);
            gradient_clipping_gpu(l.weight_updates_gpu, l.nweights, l.clip * net.batch, norm);
        }
    }

    if(net.delta_gpu){
        
        if(l.weight_transform.type) swap_weight_transform(&l);
        cudnnConvolutionBackwardData(cudnn_handle(),
                &(l.impact),
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);

        if(l.weight_transform.type) swap_weight_transform(&l);
    }
}

void pull_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float decay = a.decay;
    int batch = a.batch;

    switch (a.optim) {
        case ADAM:
            adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
            adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
            if(l.scales_gpu){
                adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
            }
            break;
            
        case SGD:
        default:
            axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
            axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
            scal_gpu(l.nweights, a.momentum, l.weight_updates_gpu, 1);

            axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
            scal_gpu(l.n, a.momentum, l.bias_updates_gpu, 1);

            if(l.scales_gpu){
                axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
                scal_gpu(l.n, a.momentum, l.scale_updates_gpu, 1);
            }
            break;
    }
    // if(l.clip) constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
}