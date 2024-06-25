#include "l2norm_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_l2norm_layer(int batch, int inputs)
{
    layer l = {0};
    l.type = L2NORM;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.output = calloc(inputs*batch, sizeof(float));
    l.scales = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward_gpu = forward_l2norm_layer_gpu;
    l.backward_gpu = backward_l2norm_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.scales_gpu = cuda_make_array(l.scales, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    return l;
}

void resize_l2norm(layer* l, int inputs) {
    l->inputs = inputs;
    l->outputs = inputs;

    l->delta = realloc(l->delta, inputs * l->batch * sizeof(float));
    l->output = realloc(l->output, inputs * l->batch * sizeof(float));
    l->scales = realloc(l->scales, inputs * l->batch * sizeof(float));

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->scales_gpu);

    l->delta_gpu = cuda_make_array(l->delta, inputs * l->batch);
    l->scales_gpu = cuda_make_array(l->scales, inputs * l->batch); 
    l->output_gpu = cuda_make_array(l->output, inputs * l->batch);
}

void forward_l2norm_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w*l.out_h);
}

void backward_l2norm_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
    axpy_gpu(l.batch*l.inputs, l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}