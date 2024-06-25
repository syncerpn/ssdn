#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"

void forward_deconvolutional_layer_gpu(layer l, network net);
void backward_deconvolutional_layer_gpu(layer l, network net);
void update_deconvolutional_layer_gpu(layer l, update_args a);
void push_deconvolutional_layer(layer l);
void pull_deconvolutional_layer(layer l);

void init_deconvolutional_layer(layer l, initializer init);
layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, activation_scheme activation, int batch_normalize, OPTIMIZER optim);
void resize_deconvolutional_layer(layer *l, int h, int w);

#endif