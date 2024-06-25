#include "normalization_layer.h"
#include "blas.h"

#include <stdio.h>

layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa)
{
    layer l = {0};
    l.type = NORMALIZATION;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.kappa = kappa;
    l.size = size;
    l.alpha = alpha;
    l.beta = beta;
    l.output = calloc(h * w * c * batch, sizeof(float));
    l.delta = calloc(h * w * c * batch, sizeof(float));
    l.squared = calloc(h * w * c * batch, sizeof(float));
    l.norms = calloc(h * w * c * batch, sizeof(float));
    l.norms_delta = calloc(h * w * c * batch, sizeof(float));

    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.forward_gpu = forward_normalization_layer_gpu;
    l.backward_gpu = backward_normalization_layer_gpu;

    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);
    l.squared_gpu = cuda_make_array(l.squared, h * w * c * batch);
    l.norms_gpu =   cuda_make_array(l.norms, h * w * c * batch);
    l.norms_delta_gpu = cuda_make_array(l.norms_delta, h * w * c * batch);

    return l;
}

void resize_normalization_layer(layer *l, int w, int h)
{
    int c = l->c;
    int batch = l->batch;
    l->h = h;
    l->w = w;
    l->out_h = h;
    l->out_w = w;
    l->inputs = w*h*c;
    l->outputs = l->inputs;
    l->output = realloc(l->output, h * w * c * batch * sizeof(float));
    l->delta = realloc(l->delta, h * w * c * batch * sizeof(float));
    l->squared = realloc(l->squared, h * w * c * batch * sizeof(float));
    l->norms = realloc(l->norms, h * w * c * batch * sizeof(float));
    l->norms_delta = realloc(l->norms_delta, h * w * c * batch * sizeof(float));

    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu); 
    cuda_free(l->squared_gpu);
    cuda_free(l->norms_gpu);
    cuda_free(l->norms_delta_gpu);

    l->output_gpu =  cuda_make_array(l->output, h * w * c * batch);
    l->delta_gpu =   cuda_make_array(l->delta, h * w * c * batch);
    l->squared_gpu = cuda_make_array(l->squared, h * w * c * batch);
    l->norms_gpu =   cuda_make_array(l->norms, h * w * c * batch);
    l->norms_delta_gpu = cuda_make_array(l->norms_delta, h * w * c * batch);
}

void forward_normalization_layer_gpu(const layer l, network net)
{
    int k,b;
    int w = l.w;
    int h = l.h;
    int c = l.c;
    scal_gpu(w*h*c*l.batch, 0, l.squared_gpu, 1);

    for(b = 0; b < l.batch; ++b){
        float *squared = l.squared_gpu + w*h*c*b;
        float *norms   = l.norms_gpu + w*h*c*b;
        float *input   = net.input_gpu + w*h*c*b;
        pow_gpu(w*h*c, 2, input, 1, squared, 1);

        const_gpu(w*h, l.kappa, norms, 1);
        for(k = 0; k < l.size/2; ++k){
            axpy_gpu(w*h, l.alpha, squared + w*h*k, 1, norms, 1);
        }

        for(k = 1; k < l.c; ++k){
            copy_gpu(w*h, norms + w*h*(k-1), 1, norms + w*h*k, 1);
            int prev = k - ((l.size-1)/2) - 1;
            int next = k + (l.size/2);
            if(prev >= 0) axpy_gpu(w*h, -l.alpha, squared + w*h*prev, 1, norms + w*h*k, 1);
            if(next < l.c) axpy_gpu(w*h,  l.alpha, squared + w*h*next, 1, norms + w*h*k, 1);
        }
    }
    pow_gpu(w*h*c*l.batch, -l.beta, l.norms_gpu, 1, l.output_gpu, 1);
    mul_gpu(w*h*c*l.batch, net.input_gpu, 1, l.output_gpu, 1);
}

void backward_normalization_layer_gpu(const layer l, network net)
{
    // TODO This is approximate ;-)

    int w = l.w;
    int h = l.h;
    int c = l.c;
    pow_gpu(w*h*c*l.batch, -l.beta, l.norms_gpu, 1, l.norms_delta_gpu, 1);
    mul_gpu(w*h*c*l.batch, l.delta_gpu, 1, l.norms_delta_gpu, 1);
    axpy_gpu(w*h*c*l.batch, l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}