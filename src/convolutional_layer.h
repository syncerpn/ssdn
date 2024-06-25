#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
#include "initializer.h"

void print_weight_transform_scheme_summary(weight_transform_scheme wts);
WEIGHT_TRANSFORM_SCHEME get_weight_transform_scheme(char *s);

void forward_convolutional_layer_gpu(layer l, network net);
void backward_convolutional_layer_gpu(layer l, network net);
void update_convolutional_layer_gpu(layer l, update_args a);

void assign_weight_transform_convolutional_layer(layer* l, weight_transform_scheme wts);
void assign_quantization_convolutional_layer(layer* l, quantization_scheme qs);

void push_convolutional_layer(layer l);
void pull_convolutional_layer(layer l);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void cudnn_convolutional_setup(layer *l);

void init_convolutional_layer(layer l, initializer init);
layer make_convolutional_layer(int batch, int h, int w, int c, int n, int dilation, int groups, int size, int stride, int padding, activation_scheme activation, int batch_normalize, OPTIMIZER optim);
void resize_convolutional_layer(layer *layer, int w, int h);
image *visualize_convolutional_layer(layer l, char *window, image *prev_weights);

image get_convolutional_image(layer l);
image get_convolutional_delta(layer l);
image get_convolutional_weight(layer l, int i);

int convolutional_out_height(layer l);
int convolutional_out_width(layer l);

//nghiant: add ref
void make_shifting_weights_max_gpu(float *weights, int n, int size, float *tran_weights, float* q_coeff, int n_coeff, int zero_center);
void make_shifting_weights_mean_gpu(float *weights, int n, int size, float *tran_weights, float* q_coeff, int n_coeff, int zero_center);
void uniform_quantize_weights_gpu(float *weights, int n, float *tran_weights, float step_size, float* q_coeff, int n_coeff, int zero_center);
void make_shifting_weights_gpu(float *weights, int n, float *tran_weights, float* q_coeff, int n_coeff, int zero_center);
void make_scale_linear_gpu(float *weights, int n, int size, float *tran_weights, int n_coeff);
void transform_weights(float *weights, int n, int size, float *tran_weights, weight_transform_scheme wts, float* q_coeff, int n_coeff);
//nghiant_end

#endif