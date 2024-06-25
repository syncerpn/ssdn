#include "logistic_layer.h"
#include "activations.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_logistic_layer(int batch, int inputs)
{
    layer l = {0};
    l.type = LOGXENT;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 

    return l;
}

void resize_logistic_layer(layer* l, int inputs) {
    l->inputs = inputs;
    l->outputs = inputs;

    l->loss = realloc(l->loss, inputs * l->batch * sizeof(float));
    l->output = realloc(l->output, inputs * l->batch * sizeof(float));
    l->delta = realloc(l->delta, inputs * l->batch * sizeof(float));

    cuda_free(l->loss_gpu);
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);

    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
    l->loss_gpu = cuda_make_array(l->loss, inputs*l->batch);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
}

void forward_logistic_layer_gpu(const layer l, network net)
{
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});
    if(net.truth){
        logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_logistic_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}