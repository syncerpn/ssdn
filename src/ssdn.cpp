#include "ssdn.h"

int main() {
	float* x = (float*)calloc(100, sizeof(float));
	float* x_gpu = cuda_make_array<float>(x, 100);
	cuda_free<float>(x_gpu);
	free(x);
}