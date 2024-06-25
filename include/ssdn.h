#ifndef SSDN_API
#define SSDN_API
#define BLOCK 512
extern int gpu_index;
extern unsigned int seed;

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"

void cuda_set_device(int n);
float *cuda_make_array(float *x, size_t n);
void cuda_free(float *x_gpu);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);

#endif