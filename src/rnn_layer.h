#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"
#define USET

void init_rnn_layer(layer l, initializer init);
layer make_rnn_layer(int batch, int inputs, int outputs, int steps, activation_scheme activation, int batch_normalize, OPTIMIZER optim);

void forward_rnn_layer_gpu(layer l, network net);
void backward_rnn_layer_gpu(layer l, network net);
void update_rnn_layer_gpu(layer l, update_args a);
void push_rnn_layer(layer l);
void pull_rnn_layer(layer l);

#endif