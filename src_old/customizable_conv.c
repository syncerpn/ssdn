#include "customizable_conv.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include <assert.h>

void init_custom_conv_layer(layer l, initializer init) {
    init.auto_sigma = sqrt(1./(l.size * l.size * l.c));
    initialize_array(l.weights, l.nweights, init);
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
}

int custom_conv_out_height(layer l) {return (l.h + 2*l.pad - l.size) / l.stride + 1;}
int custom_conv_out_width(layer l) {return (l.w + 2*l.pad - l.size) / l.stride + 1;}

layer make_custom_conv_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, activation_scheme activation)
{
    layer l = {0};
    l.type = CONVOLUTIONAL;

    l.w = w;
    l.h = h;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;

    l.weights = calloc(c*n*size*size, sizeof(float));
    l.biases = calloc(n, sizeof(float));
    l.nweights = c * n * size * size;
    l.nbiases = n;

    int out_w = custom_conv_out_width(l);
    int out_h = custom_conv_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*l.outputs, sizeof(float));

    l.forward_gpu = forward_custom_conv_layer_gpu;

    l.weights_gpu = cuda_make_array(l.weights, l.nweights);
    l.biases_gpu = cuda_make_array(l.biases, n);

    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

    l.input_im = calloc(l.batch * l.out_w * l.out_h * l.size * l.size * l.c, sizeof(float));
    l.input_im_gpu = cuda_make_array(l.input_im, l.batch * l.out_w * l.out_h * l.size * l.size * l.c);
    
    l.activation = activation;
    return l;
}

void resize_custom_conv_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    int out_w = custom_conv_out_width(*l);
    int out_h = custom_conv_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    cuda_free(l->output_gpu);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    l->input_im = realloc(l->input_im, l->batch * l->out_w * l->out_h * l->size * l->size * l->c * sizeof(float));
    cuda_free(l->input_im_gpu);
    l->input_im_gpu = cuda_make_array(l->input_im, l->batch * l->out_w * l->out_h * l->size * l->size * l->c);
}