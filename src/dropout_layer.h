#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

layer make_dropout_layer(int batch, int inputs, float probability);
void resize_dropout_layer(layer *l, int inputs);
void forward_dropout_layer_gpu(layer l, network net);
void backward_dropout_layer_gpu(layer l, network net);

#endif