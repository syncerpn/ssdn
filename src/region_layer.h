#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_region_layer(int batch, int w, int h, int n, int classes, int max_boxes, float* anchors, int softmax, int background);
void resize_region_layer(layer *l, int w, int h);

void forward_region_layer(const layer l, network net);
void forward_region_layer_gpu(const layer l, network net);
void backward_region_layer_gpu(layer l, network net);

#endif