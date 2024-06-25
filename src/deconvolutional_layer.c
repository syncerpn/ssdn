#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"

#include <stdio.h>
#include <time.h>

static size_t get_workspace_size(layer l){
    return (size_t)l.h*l.w*l.size*l.size*l.n*sizeof(float);
}

//nghiant: comment out
void bilinear_init(layer l)
{
    int i,j,f;
    float center = (l.size-1) / 2.;
    for(f = 0; f < l.n; ++f){
        for(j = 0; j < l.size; ++j){
            for(i = 0; i < l.size; ++i){
                float val = (1 - fabs(i - center)) * (1 - fabs(j - center));
                int c = f%l.c;
                int ind = f*l.size*l.size*l.c + c*l.size*l.size + j*l.size + i;
                l.weights[ind] = val;
            }
        }
    }
}

void init_deconvolutional_layer(layer l, initializer init) {
    init.auto_sigma = 0.02; //nghiant: we might want to change this shit
    initialize_array(l.weights, l.nweights, init);
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
}

layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, activation_scheme activation, int batch_normalize, OPTIMIZER optim)
{
    int i;
    layer l = {0};
    l.type = DECONVOLUTIONAL;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;

    l.nweights = c*n*size*size;
    l.nbiases = n;

    //nghiant_20190228: please add support for group convolution
    l.groups = 1;
    //nghiant_20190228_end

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.weight_updates = calloc(c*n*size*size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    for(i = 0; i < n; ++i){
        l.biases[i] = 0;
    }
    l.pad = padding;

    l.out_h = (l.h - 1) * l.stride + l.size - 2*l.pad;
    l.out_w = (l.w - 1) * l.stride + l.size - 2*l.pad;
    l.out_c = n;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;

    scal_cpu(l.nweights, (float)l.out_w*l.out_h/(l.w*l.h), l.weights, 1);

    l.output = calloc(l.batch*l.outputs, sizeof(float));
    l.delta  = calloc(l.batch*l.outputs, sizeof(float));

    l.batch_normalize = batch_normalize;

    if(batch_normalize){
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
    
    if(optim == ADAM){
        l.m = calloc(c*n*size*size, sizeof(float));
        l.v = calloc(c*n*size*size, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));

        l.m_gpu = cuda_make_array(l.m, c*n*size*size);
        l.v_gpu = cuda_make_array(l.v, c*n*size*size);
        l.bias_m_gpu = cuda_make_array(l.bias_m, n);
        l.bias_v_gpu = cuda_make_array(l.bias_v, n);
        l.scale_m_gpu = cuda_make_array(l.scale_m, n);
        l.scale_v_gpu = cuda_make_array(l.scale_v, n);
    }

    l.forward_gpu = forward_deconvolutional_layer_gpu;
    l.backward_gpu = backward_deconvolutional_layer_gpu;

    l.update_gpu = update_deconvolutional_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, c*n*size*size);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.delta_gpu = cuda_make_array(l.delta, l.batch*l.out_h*l.out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*l.out_h*l.out_w*n);

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(0, n);
        l.variance_gpu = cuda_make_array(0, n);

        l.rolling_mean_gpu = cuda_make_array(0, n);
        l.rolling_variance_gpu = cuda_make_array(0, n);

        l.mean_delta_gpu = cuda_make_array(0, n);
        l.variance_delta_gpu = cuda_make_array(0, n);

        l.scales_gpu = cuda_make_array(l.scales, n);
        l.scale_updates_gpu = cuda_make_array(0, n);

        l.x_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
        l.x_norm_gpu = cuda_make_array(0, l.batch*l.out_h*l.out_w*n);
    }

    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 

    l.activation = activation;
    l.workspace_size = get_workspace_size(l);

    return l;
}

void denormalize_deconvolutional_layer(layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c*l.size*l.size; ++j){
            l.weights[i*l.c*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

void resize_deconvolutional_layer(layer *l, int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = (l->h - 1) * l->stride + l->size - 2*l->pad;
    l->out_w = (l->w - 1) * l->stride + l->size - 2*l->pad;

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
        cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w); 
        cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1); 
    l->workspace_size = get_workspace_size(*l);
}