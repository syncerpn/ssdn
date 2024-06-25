#include "sr_flat_layer.h"
#include "cuda.h"
#include "blas.h"
#include <assert.h>
#include <stdio.h>

layer make_sr_flat_layer(int batch, int w, int h, int c, float scale)
{
    layer l = {0};
    l.type = SR_FLAT;
    l.batch = batch;
    if (scale != 0) {
        float r = floorf(c/scale/scale);
        assert(r*scale*scale == c);
        l.sr_flat_scale = scale;
    } else {
        float sc = floorf(sqrt(c));
        assert(sc*sc == c);
        l.sr_flat_scale = sc;
    }
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*(int)(l.sr_flat_scale);
    l.out_h = h*(int)(l.sr_flat_scale);
    l.out_c = (int)(l.c / l.sr_flat_scale / l.sr_flat_scale);
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward_gpu = forward_sr_flat_layer_gpu;
    l.backward_gpu = backward_sr_flat_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);

    return l;
}

void resize_sr_flat_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->out_w = w*(int)(l->sr_flat_scale);
    l->out_h = h*(int)(l->sr_flat_scale);

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
}

void forward_sr_flat_layer_gpu(const layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    sr_flat_gpu(net.input_gpu, l.w, l.h, l.c, l.sr_flat_scale, l.batch, 1, 1, l.output_gpu);
}

void backward_sr_flat_layer_gpu(const layer l, network net)
{
    sr_flat_gpu(net.delta_gpu, l.w, l.h, l.c, l.sr_flat_scale, l.batch, 0, l.impact, l.delta_gpu);
    // int i;
    // cuda_pull_array(l.delta_gpu, l.delta, l.outputs);
    // for (i = 0; i < l.outputs; ++i) {
    //     fprintf(stderr, "ldelta %.6f\n", l.delta[i]);
    // }

    // cuda_pull_array(net.delta_gpu, l.delta, l.outputs);
    // for (i = 0; i < l.outputs; ++i) {
    //     fprintf(stderr, "ndelta %.6f\n", l.delta[i]);
    // }
}