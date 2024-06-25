#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

layer make_activation_layer(int batch, int inputs, activation_scheme activation);
void resize_activation_layer(layer* l, int inputs);
void forward_activation_layer_gpu(layer l, network net);
void backward_activation_layer_gpu(layer l, network net);

#endif