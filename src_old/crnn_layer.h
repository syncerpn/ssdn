#ifndef CRNN_LAYER_H
#define CRNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"

void init_crnn_layer(layer l, initializer init);
layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, activation_scheme activation, int batch_normalize, OPTIMIZER optim);
void forward_crnn_layer_gpu(layer l, network net);
void backward_crnn_layer_gpu(layer l, network net);
void update_crnn_layer_gpu(layer l, update_args a);
void push_crnn_layer(layer l);
void pull_crnn_layer(layer l);

#endif