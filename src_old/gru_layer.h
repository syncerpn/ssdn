#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

void init_gru_layer(layer l, initializer init);
layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, OPTIMIZER optim);
void forward_gru_layer_gpu(layer l, network state);
void backward_gru_layer_gpu(layer l, network state);
void update_gru_layer_gpu(layer l, update_args a);
void push_gru_layer(layer l);
void pull_gru_layer(layer l);

#endif