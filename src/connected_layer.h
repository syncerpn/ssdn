#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"

void init_connected_layer(layer l, initializer init);
layer make_connected_layer(int batch, int inputs, int outputs, activation_scheme activation, int batch_normalize, OPTIMIZER optim);
void forward_connected_layer_gpu(layer l, network net);
void backward_connected_layer_gpu(layer l, network net);
void update_connected_layer_gpu(layer l, update_args a);
void push_connected_layer(layer l);
void pull_connected_layer(layer l);

#endif