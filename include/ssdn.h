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

#endif