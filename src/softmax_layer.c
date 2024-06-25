#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    layer l = {0};
    //nghiant_20190819: add w, h, c
    l.w = inputs;
    l.h = 1;
    l.c = 1;
    l.out_w = inputs;
    l.out_h = 1;
    l.out_c = 1;
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 

    return l;
}

void resize_softmax_layer(layer* l, int inputs) {
    l->inputs = inputs;
    l->w = inputs;
    l->outputs = inputs;
    l->out_w = inputs;

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

void pull_softmax_layer_output(const layer l)
{
    cuda_pull_array(l.output_gpu, l.output, l.inputs*l.batch);
}

void forward_softmax_layer_gpu(const layer l, network net)
{
    if(l.softmax_tree){
        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
    } else {
        if(l.spatial){
            softmax_gpu(net.input_gpu, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
            softmax_gpu(net.input_gpu, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
        }
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
        if(l.softmax_tree){
            mask_gpu(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
            mask_gpu(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}