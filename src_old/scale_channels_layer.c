#include "scale_channels_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_scale_channels_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2, int scale_wh)
{
    layer l = {0};
    l.type = SCALE_CHANNELS;
    l.batch = batch;
    l.scale_wh = scale_wh;
    l.w = w;
    l.h = h;
    l.c = c;
    if (!l.scale_wh) assert(w == 1 && h == 1);
    else assert(c == 1);

    l.out_w = w2;
    l.out_h = h2;
    l.out_c = c2;
    if (!l.scale_wh) assert(l.out_c == l.c);
    else assert(l.out_w == l.w && l.out_h == l.h);

    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.outputs;
    l.index = index;

    l.delta = calloc(l.outputs * batch, sizeof(float));
    l.output = calloc(l.outputs * batch, sizeof(float));

    l.forward_gpu = forward_scale_channels_layer_gpu;
    l.backward_gpu = backward_scale_channels_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    return l;
}

void resize_scale_channels_layer(layer *l, network *net)
{
    layer first = net->layers[l->index];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->outputs;
    l->delta = realloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = realloc(l->output, l->outputs * l->batch * sizeof(float));

    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
}