#include "sam_layer.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>

layer make_sam_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    layer l = {0};
    l.type = SAM;

    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;

    l.out_w = w2;
    l.out_h = h2;
    l.out_c = c2;
    
    assert(l.out_c == l.c);
    assert(l.w == l.out_w && l.h == l.out_h);

    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.outputs;
    l.index = index;

    l.delta = calloc(l.outputs * batch, sizeof(float));
    l.output = calloc(l.outputs * batch, sizeof(float));

    l.forward_gpu = forward_sam_layer_gpu;
    l.backward_gpu = backward_sam_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    return l;
}

void resize_sam_layer(layer *l, int w, int h)
{
    l->out_w = w;
    l->out_h = h;
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->outputs;
    l->delta = realloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = realloc(l->output, l->outputs * l->batch * sizeof(float));

    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
}