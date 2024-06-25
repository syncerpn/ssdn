#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

image get_maxpool_image(layer l);
layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding, int dilation, int spatial);
void resize_maxpool_layer(layer *l, int w, int h);

void forward_maxpool_layer_gpu(layer l, network net);
void backward_maxpool_layer_gpu(layer l, network net);
#endif