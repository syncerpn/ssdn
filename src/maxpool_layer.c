#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.output);
}

image get_maxpool_delta(layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return float_to_image(w,h,c,l.delta);
}

layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding, int dilation, int spatial)
{
    layer l = {0};
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.spatial = spatial;

    if (spatial) {
        l.pad = 0;
        l.dilation = 1;
        l.size = -1;
        l.stride = 1;
        l.out_w = 1;
        l.out_h = 1;
    } else {
        l.pad = padding;
        l.dilation = dilation;
        l.size = size;
        l.stride = stride;
        l.out_w = (w + padding - size - (size-1)*(dilation-1))/stride + 1;
        l.out_h = (h + padding - size - (size-1)*(dilation-1))/stride + 1;
    }
    
    l.out_c = c;

    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h * w * c;
    int output_size = l.outputs * batch;

    l.indexes = calloc(output_size, sizeof(float));
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));

    l.indexes_gpu = cuda_make_int_array(0, output_size);
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);

    l.forward_gpu = forward_maxpool_layer_gpu;
    l.backward_gpu = backward_maxpool_layer_gpu;

    return l;
}

void resize_maxpool_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;

    if (!l->spatial) {
        l->out_w = (w + l->pad - l->size - (l->size - 1) * (l->dilation - 1)) / l->stride + 1;
        l->out_h = (h + l->pad - l->size - (l->size - 1) * (l->dilation - 1)) / l->stride + 1;
        l->outputs = l->out_w * l->out_h * l->c;
        int output_size = l->outputs * l->batch;

        l->indexes = realloc(l->indexes, output_size * sizeof(int));
        l->output = realloc(l->output, output_size * sizeof(float));
        l->delta = realloc(l->delta, output_size * sizeof(float));

        cuda_free((float *)l->indexes_gpu);
        cuda_free(l->output_gpu);
        cuda_free(l->delta_gpu);
        
        l->indexes_gpu = cuda_make_int_array(0, output_size);
        l->output_gpu  = cuda_make_array(l->output, output_size);
        l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    }
}