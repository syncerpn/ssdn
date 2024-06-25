#ifndef UPSAMPLE_LAYER_H
#define UPSAMPLE_LAYER_H
#include "darknet.h"

layer make_upsample_layer(int batch, int w, int h, int c, int stride);
void resize_upsample_layer(layer *l, int w, int h);

void forward_upsample_layer_gpu(const layer l, network net);
void backward_upsample_layer_gpu(const layer l, network net);

#endif