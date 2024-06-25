#ifndef CUSTOM_CONV_LAYER_H
#define CUSTOM_CONV_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"

void forward_custom_conv_layer_gpu(layer l, network net);

void push_custom_conv_layer(layer l);
void pull_custom_conv_layer(layer l);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);

void init_custom_conv_layer(layer l, initializer init);
layer make_custom_conv_layer(int batch, int h, int w, int c, int n, int size, int stride, int padding, activation_scheme activation);
void resize_custom_conv_layer(layer *layer, int w, int h);

int custom_conv_out_height(layer l);
int custom_conv_out_width(layer l);

#endif