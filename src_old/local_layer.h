#ifndef LOCAL_LAYER_H
#define LOCAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"

void forward_local_layer_gpu(layer l, network net);
void backward_local_layer_gpu(layer l, network net);

void update_local_layer_gpu(layer l, update_args a);

void push_local_layer(layer l);
void pull_local_layer(layer l);

void init_local_layer(layer l, initializer init);
layer make_local_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, activation_scheme activation, OPTIMIZER optim);

#endif