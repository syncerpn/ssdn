#ifndef STACK_LAYER_H
#define STACK_LAYER_H
#include "darknet.h"

layer make_stack_layer(int batch, int w, int h, int c, int stride);
void resize_stack_layer(layer *l, int w, int h);

void forward_stack_layer_gpu(const layer l, network net);
void backward_stack_layer_gpu(const layer l, network net);

#endif