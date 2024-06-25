#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "norm_w_convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "quantization_layer.h"
}

void norm_w_transform_weights(float *weights, int n, int size, float *tran_weights, weight_transform_scheme wts, float* q_coeff, int n_coeff)
{
    switch(wts.type){
        case WTS_MAX_SHIFTER:
            norm_w_make_shifting_weights_max_gpu(weights, n, size, tran_weights, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_MEAN_SHIFTER:
            norm_w_make_shifting_weights_mean_gpu(weights, n, size, tran_weights, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_SHIFTER:
            norm_w_make_shifting_weights_gpu(weights, n * size, tran_weights, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_UNIFORM:
            norm_w_uniform_quantize_weights_gpu(weights, n * size, tran_weights, wts.step_size, q_coeff, n_coeff, wts.num_level % 2);
            break;
        case WTS_SCALE_LINEAR:
            norm_w_make_scale_linear_gpu(weights, n, size, tran_weights, n_coeff);
            break;
        case WTS_NONE:
            break;
    }
}

void forward_norm_w_convolutional_layer_gpu(layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

	if (l.weight_transform.type) {
		norm_w_normalize_gpu(l.weights_v_gpu, l.weights_v_norm_gpu, l.nweights_v/l.n, l.n);
		norm_w_norm_div_weights_gpu(l.weights_gpu, l.weights_v_gpu, l.weights_v_norm_gpu, l.nweights_v/l.n, l.n);
		norm_w_transform_weights(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.tran_weights_gpu, l.weight_transform, l.q_coeff_gpu, l.n_coeff);

		norm_w_scale_g_weights_gpu(l.weights_gpu, l.weights_g_gpu, l.nweights_v/l.n, l.n);
		norm_w_scale_g_weights_gpu(l.tran_weights_gpu, l.weights_g_gpu, l.nweights_v/l.n, l.n);
		swap_weight_transform(&l);
	} else {
    //nghiant: norm_w stuff
	    norm_w_normalize_gpu(l.weights_v_gpu, l.weights_v_norm_gpu, l.nweights_v/l.n, l.n);
	    norm_w_reform_weights_gpu(l.weights_gpu, l.weights_v_gpu, l.weights_v_norm_gpu, l.weights_g_gpu, l.nweights_v/l.n, l.n);
    }
    //nghiant_end

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

void backward_norm_w_convolutional_layer_gpu(layer l, network net)
{
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

    norm_w_delta_g_gpu(l.weight_updates_g_gpu, l.weight_updates_gpu, l.weights_v_gpu, l.weights_v_norm_gpu, l.nweights_v/l.n, l.n);
    norm_w_delta_v_gpu(l.weight_updates_v_gpu, l.weight_updates_gpu, l.weight_updates_g_gpu, l.weights_v_gpu, l.weights_v_norm_gpu, l.weights_g_gpu, l.nweights_v/l.n, l.n);

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

void pull_norm_w_convolutional_layer(layer l)
{
    cuda_pull_array(l.weights_v_gpu, l.weights_v, l.nweights_v);
    cuda_pull_array(l.weights_g_gpu, l.weights_g, l.nweights_g);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);

    cuda_pull_array(l.weight_updates_v_gpu, l.weight_updates_v, l.nweights_v);
    cuda_pull_array(l.weight_updates_g_gpu, l.weight_updates_g, l.nweights_g);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void push_norm_w_convolutional_layer(layer l)
{
    cuda_push_array(l.weights_v_gpu, l.weights_v, l.nweights_v);
    cuda_push_array(l.weights_g_gpu, l.weights_g, l.nweights_g);
    cuda_push_array(l.biases_gpu, l.biases, l.n);

    cuda_push_array(l.weight_updates_v_gpu, l.weight_updates_v, l.nweights_v);
    cuda_push_array(l.weight_updates_g_gpu, l.weight_updates_g, l.nweights_g);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
    if (l.batch_normalize){
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_norm_w_convolutional_layer_gpu(layer l, update_args a)
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
            axpy_gpu(l.nweights_v, -decay*batch, l.weights_v_gpu, 1, l.weight_updates_v_gpu, 1);
            axpy_gpu(l.nweights_v, learning_rate/batch, l.weight_updates_v_gpu, 1, l.weights_v_gpu, 1);
            scal_gpu(l.nweights_v, a.momentum, l.weight_updates_v_gpu, 1);

            axpy_gpu(l.n, learning_rate/batch, l.weight_updates_g_gpu, 1, l.weights_g_gpu, 1);
            scal_gpu(l.n, a.momentum, l.weight_updates_g_gpu, 1);

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