#include "ssdn.h"

int main() {
	float* x = (float*)calloc(100, sizeof(float));
	float* x_gpu = cuda_make_array(x, 100);
	cuda_free(x_gpu);
	free(x);
}