#include "dropout_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

layer make_dropout_layer(int batch, int inputs, float probability)
{
    layer l = {0};
    l.type = DROPOUT;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(inputs*batch, sizeof(float));
    l.scale = 1./(1.-probability);

    l.forward_gpu = forward_dropout_layer_gpu;
    l.backward_gpu = backward_dropout_layer_gpu;
    
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);

    l.output = calloc(batch * inputs, sizeof(float));
    l.output_gpu = cuda_make_array(l.output, batch * inputs);
    l.delta = calloc(batch * inputs, sizeof(float));
    l.delta_gpu = cuda_make_array(l.delta, batch * inputs);

    return l;
} 

void resize_dropout_layer(layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;

    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, l->inputs*l->batch);

    l->output = realloc(l->output, l->batch * l->inputs * sizeof(float));
    l->delta = realloc(l->delta, l->batch * l->inputs * sizeof(float));

    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);

    l->output_gpu = cuda_make_array(l->output, l->batch * l->inputs);
    l->delta_gpu = cuda_make_array(l->delta, l->batch * l->inputs);
}