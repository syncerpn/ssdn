#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"

layer make_route_layer(int batch, int n, int *input_layers, int *input_size);
void resize_route_layer(layer *l, network *net);

void forward_route_layer_gpu(const layer l, network net);
void backward_route_layer_gpu(const layer l, network net);

#endif