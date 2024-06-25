#include "activation_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_activation_layer(int batch, int inputs, activation_scheme activation)
{
    layer l = {0};
    l.type = ACTIVE;

    l.inputs = inputs;
    l.outputs = inputs;
    l.batch=batch;

    l.output = calloc(batch*inputs, sizeof(float*));
    l.delta = calloc(batch*inputs, sizeof(float*));

    l.forward_gpu = forward_activation_layer_gpu;
    l.backward_gpu = backward_activation_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);

    l.activation = activation;
    return l;
}

void resize_activation_layer(layer* l, int inputs) {
    l->inputs = inputs;
    l->outputs = inputs;

    l->output = realloc(l->output, l->batch * inputs * sizeof(float));
    l->delta = realloc(l->delta, l->batch * inputs * sizeof(float));

    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);

    l->output_gpu = cuda_make_array(l->output, l->batch * inputs);
    l->delta_gpu = cuda_make_array(l->delta, l->batch * inputs);
}

void forward_activation_layer_gpu(layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_activation_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs * l.batch, l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}