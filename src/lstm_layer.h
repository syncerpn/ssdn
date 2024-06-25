#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"
#define USET

void init_lstm_layer(layer l, initializer init);
layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, OPTIMIZER optim);

void forward_lstm_layer_gpu(layer l, network net);
void backward_lstm_layer_gpu(layer l, network net);
void update_lstm_layer_gpu(layer l, update_args a); 

#endif