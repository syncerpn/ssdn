#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

ACTIVATION get_activation(char *s);
char *get_activation_string(ACTIVATION a);
char *get_activation_string_cap(ACTIVATION a);
void print_activation_summary(activation_scheme act);
void activate_array_gpu(float *x, int n, activation_scheme a);
void gradient_array_gpu(float *x, int n, activation_scheme a, float *delta);

#endif