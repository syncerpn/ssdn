#ifndef AVGPOOL_LAYER_H
#define AVGPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

image get_avgpool_image(layer l);
layer make_avgpool_layer(int batch, int w, int h, int c, int size, int stride, int padding, int dilation, int spatial);
void resize_avgpool_layer(layer *l, int w, int h);

void forward_avgpool_layer_gpu(layer l, network net);
void backward_avgpool_layer_gpu(layer l, network net);

#endif