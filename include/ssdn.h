#define BLOCK 512
extern int gpu_index;

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cudnn.h"

void cuda_set_device(int n);
template <typename T> T *cuda_make_array(T *x, size_t n);
template <typename T> void cuda_free(T *x_gpu);
template <typename T> void cuda_push_array(T *x_gpu, T *x, size_t n);
template <typename T> void cuda_pull_array(T *x_gpu, T *x, size_t n);