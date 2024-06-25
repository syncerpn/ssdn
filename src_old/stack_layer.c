#include "stack_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

layer make_stack_layer(int batch, int w, int h, int c, int stride)
{
    layer l = {0};
    l.type = STACK;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c*stride;
    if(stride < 0){
        stride = -stride;
        l.reverse=1;
        l.out_c = c/stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward_gpu = forward_stack_layer_gpu;
    l.backward_gpu = backward_stack_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);

    return l;
}

void resize_stack_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->out_w = w;
    l->out_h = h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
}


void forward_stack_layer_gpu(const layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.reverse){
        stack_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu);
    }else{
        stack_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
    }
}

void backward_stack_layer_gpu(const layer l, network net)
{
    if(l.reverse){
        stack_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale * l.impact, net.delta_gpu);
    }else{
        stack_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale * l.impact, l.delta_gpu);
    }
}