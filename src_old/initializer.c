#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "initializer.h"
#include "utils.h"

INITIALIZER get_initializer_type(char* type) {
    if (strcmp(type, "none")==0) return NONE;
    if (strcmp(type, "xavier")==0) return XAVIER;
    if (strcmp(type, "he")==0) return HE;
    if (strcmp(type, "normal")==0) return NORMAL;
    if (strcmp(type, "scalar")==0) return SCALAR;
    if (strcmp(type, "uniform")==0) return UNIFORM;
    return NONE;
}

char* get_initializer_string(INITIALIZER type) {
    switch (type) {
        case XAVIER:
            return "xavier";
        case HE:
            return "he";
        case NORMAL:
            return "normal";
        case SCALAR:
            return "scalar";
        case UNIFORM:
            return "uniform";
        case NONE:
        default:
            return "none";
    }
}

void xavier_initialize_array(float* x, int nx, float sigma) {
	int i;
	for (i = 0; i < nx; ++i) {
		x[i] = sigma * rand_normal();
	}
	return;
}

void he_initialize_array(float* x, int nx, float sigma) {
	int i;
	for (i = 0; i < nx; ++i) {
		x[i] = sqrt(2.) * sigma * rand_normal();
	}
	return;
}

void normal_initialize_array(float* x, int nx, float mu, float sigma) {
    int i;
    for (i = 0; i < nx; ++i) {
        x[i] = sigma * rand_normal() + mu;
    }
    return;
}

void scalar_initialize_array(float* x, int nx, float alpha) {
    int i;
    for (i = 0; i < nx; ++i) {
        x[i] = alpha;
    }
    return;
}

void uniform_initialize_array(float* x, int nx, float sigma) {
    int i;
    for (i = 0; i < nx; ++i) {
        x[i] = sqrt(2.) * sigma * rand_uniform(-1,1);
    }
    return;
}

void initialize_array(float* x, int nx, initializer init) {
	switch (init.type) {
		case XAVIER:
			xavier_initialize_array(x, nx, init.auto_sigma);
			return;
		case HE:
			he_initialize_array(x, nx, init.auto_sigma);
			return;
        case NORMAL:
            normal_initialize_array(x, nx, init.mu, init.sigma);
            return;
        case SCALAR:
            scalar_initialize_array(x, nx, init.alpha);
            return;
        case UNIFORM:
            uniform_initialize_array(x, nx, init.auto_sigma);
            return;
		case NONE:
		default:
			return;
	}
}