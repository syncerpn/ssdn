#ifndef QUANTIZATION_LAYER_H
#define QUANTIZATION_LAYER_H

#include "cuda.h"
#include "layer.h"
#include "network.h"
#include "darknet.h"

void print_quantization_scheme_summary(quantization_scheme qs);
QUANTIZATION_SCHEME get_quantization_scheme(char *s);
layer make_quantization_layer(int batch, int w, int h, int c, quantization_scheme qs);
void resize_quantization_layer(layer* l, int w, int h);
void forward_quantization_layer_gpu(layer l, network net);
void backward_quantization_layer_gpu(layer l, network net);
void quantize_array_forward_gpu(float* x, int n, quantization_scheme qs);
void quantize_array_backward_gpu(float* x, int n, quantization_scheme qs, float* delta);

#endif