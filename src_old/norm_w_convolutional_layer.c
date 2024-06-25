#include "norm_w_convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include <assert.h>

void init_norm_w_convolutional_layer(layer l, initializer init) {
    init.auto_sigma = sqrt(1./(l.size * l.size * l.c / l.groups));
    initialize_array(l.weights_v, l.nweights_v, init);

    int i;
    for (i = 0; i < l.nweights_g; ++i) {
        l.weights_g[i] = 1;
    }

    cuda_push_array(l.weights_v_gpu, l.weights_v, l.nweights_v);
    cuda_push_array(l.weights_g_gpu, l.weights_g, l.nweights_g);
}

int norm_w_convolutional_out_height(layer l)
{
    return (l.h + 2*l.pad - l.size - (l.dilation - 1) * (l.size - 1)) / l.stride + 1; //nghiant: re-calculate with dilation
}

int norm_w_convolutional_out_width(layer l)
{
    return (l.w + 2*l.pad - l.size - (l.dilation - 1) * (l.size - 1)) / l.stride + 1; //nghiant: re-calculate with dilation
}

static size_t get_workspace_size(layer l){
    size_t most = 0;
    size_t s = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.weightDesc,
            l.convDesc,
            l.dstTensorDesc,
            l.fw_algo,
            &s);
    if (s > most) most = s;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dweightDesc,
            l.bf_algo,
            &s);
    if (s > most) most = s;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
            l.weightDesc,
            l.ddstTensorDesc,
            l.convDesc,
            l.dsrcTensorDesc,
            l.bd_algo,
            &s);
    if (s > most) most = s;
    return most;
}

void cudnn_norm_w_convolutional_setup(layer *l)
{
    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    #else
    cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
    cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
    if(l->groups > 1){
        error("CUDNN < 7 doesn't support groups, please upgrade!");
    }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}

layer make_norm_w_convolutional_layer(int batch, int h, int w, int c, int n, int dilation, int groups, int size, int stride, int padding, activation_scheme activation, int batch_normalize, OPTIMIZER optim)
{
    int i;
    layer l = {0};
    l.type = NORM_W_CONVOLUTIONAL;

    l.dilation = dilation;
    l.groups = groups;
    l.w = w;
    l.h = h;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.size = size;

    //nghiant: norm_w stuff
    l.nweights_v = c/groups*n*size*size;
    l.nweights_g = n;

    l.weights_v = calloc(l.nweights_v, sizeof(float));
    l.weights_g = calloc(l.nweights_g, sizeof(float));

    l.weights_v_gpu = cuda_make_array(l.weights_v, l.nweights_v);
    l.weights_g_gpu = cuda_make_array(l.weights_g, l.nweights_g);

    l.weights_v_norm = calloc(n, sizeof(float));
    l.weights_v_norm_gpu = cuda_make_array(l.weights_v_norm, n);

    l.weight_updates_v = calloc(l.nweights_v, sizeof(float));
    l.weight_updates_g = calloc(l.nweights_g, sizeof(float));

    l.weight_updates_v_gpu = cuda_make_array(l.weight_updates_v, l.nweights_v);
    l.weight_updates_g_gpu = cuda_make_array(l.weight_updates_g, l.nweights_g);
    //nghiant_end: norm_w stuff

    l.weights = calloc(c/groups*n*size*size, sizeof(float));
    l.weight_updates = calloc(c/groups*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    int out_w = norm_w_convolutional_out_width(l);
    int out_h = norm_w_convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    if (batch_normalize) {
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = calloc(l.batch*l.outputs, sizeof(float));
    }
    
    if (optim == ADAM) {
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
        
        l.m_gpu = cuda_make_array(l.m, l.nweights);
        l.v_gpu = cuda_make_array(l.v, l.nweights);
        l.bias_m_gpu = cuda_make_array(l.bias_m, n);
        l.bias_v_gpu = cuda_make_array(l.bias_v, n);
        l.scale_m_gpu = cuda_make_array(l.scale_m, n);
        l.scale_v_gpu = cuda_make_array(l.scale_v, n);
    }

    l.forward_gpu = forward_norm_w_convolutional_layer_gpu;
    l.backward_gpu = backward_norm_w_convolutional_layer_gpu;

    l.update_gpu = update_norm_w_convolutional_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, l.nweights);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, n);
        l.variance_gpu = cuda_make_array(l.variance, n);

        l.rolling_mean_gpu = cuda_make_array(l.mean, n);
        l.rolling_variance_gpu = cuda_make_array(l.variance, n);

        l.mean_delta_gpu = cuda_make_array(l.mean, n);
        l.variance_delta_gpu = cuda_make_array(l.variance, n);

        l.scales_gpu = cuda_make_array(l.scales, n);
        l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

        l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    }
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnCreateTensorDescriptor(&l.srcTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnCreateFilterDescriptor(&l.weightDesc);
    cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
    cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
    cudnnCreateFilterDescriptor(&l.dweightDesc);
    cudnnCreateConvolutionDescriptor(&l.convDesc);
    cudnn_norm_w_convolutional_setup(&l);
    
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;
    return l;
}

void resize_norm_w_convolutional_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = norm_w_convolutional_out_width(*l);
    int out_h = norm_w_convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));
    if(l->batch_normalize){
        l->x = realloc(l->x, l->batch*l->outputs*sizeof(float));
        l->x_norm  = realloc(l->x_norm, l->batch*l->outputs*sizeof(float));
    }

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if(l->batch_normalize){
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }

    cudnn_norm_w_convolutional_setup(l);
    l->workspace_size = get_workspace_size(*l);
}

void assign_weight_transform_norm_w_convolutional_layer(layer* l, weight_transform_scheme wts) {
    l->weight_transform = wts;

    if (l->tran_weights == 0) l->tran_weights = calloc(l->nweights, sizeof(float));
    if (l->tran_weights_gpu == 0) l->tran_weights_gpu = cuda_make_array(l->tran_weights, l->nweights);

    l->n_coeff = wts.num_level / 2;
    
    if (l->q_coeff) {
        free(l->q_coeff);
        l->q_coeff = 0;
    }
    if (l->q_coeff_gpu) {
        cuda_free(l->q_coeff_gpu);
        l->q_coeff_gpu = 0;
    }
    if (l->n_coeff > 0) l->q_coeff = calloc(l->n_coeff, sizeof(float));

    int i;
    // float sl_step_size;
    switch (wts.type) {
        case WTS_MAX_SHIFTER:
        case WTS_MEAN_SHIFTER:
        case WTS_SHIFTER:
            for (i = 0; i < l->n_coeff; ++i) {
                l->q_coeff[i] = (float)(wts.first_shifting_factor - i);
            }
            break;
        case WTS_UNIFORM:
            if (wts.num_level % 2 == 1) {
                for (i = 0; i < l->n_coeff; ++i) {
                    l->q_coeff[i] = wts.step_size * (i+1);
                }
            } else {
                for (i = 0; i < l->n_coeff; ++i) {
                    l->q_coeff[i] = wts.step_size * (i+0.5);
                }
            }
            break;
        case WTS_SCALE_LINEAR:
            // sl_step_size = 1./l->n_coeff;
            // l->weight_transform.step_size = sl_step_size;
            // for (i = 0; i < l->n_coeff; ++i) {
            //     l->q_coeff[i] = sl_step_size * (i+1);
            // }
            break;
        case WTS_NONE:
            break;
    }

    if (l->n_coeff > 0) l->q_coeff_gpu = cuda_make_array(l->q_coeff, l->n_coeff);
}

void assign_quantization_norm_w_convolutional_layer(layer* l, quantization_scheme qs) {
    l->quantization = qs;
}