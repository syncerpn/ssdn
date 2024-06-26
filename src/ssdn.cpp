#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "ssdn.h"

void lead_one_encode(int x, int &k, int &p) {
	if (x == 0 || x == 1) {
		k = 0;
		p = 0;
	}
	k = 0;
	int xx = x;
	while (xx > 1) {
		xx = xx >> 1;
		k += 1;
	}
	p = x - (1 << k);
}

int approximate(int i, int j) {
	if (i == 0 || j == 0) return 0;

	int sign_i = i > 0 ? 1 : -1;
	int sign_j = j > 0 ? 1 : -1;

	i = sign_i * i;
	j = sign_j * j;

	int ki, pi, kj, pj;
	lead_one_encode(i, ki, pi);
	lead_one_encode(j, kj, pj);

	int p1 = pi * pj;
	int p2 = ((1 << ki) - pi) * ((1 << kj) - pj);
	int e = p1 < p2 ? p1 : p2;
	return sign_i * sign_j * (i * j - e);
}

float quantize(float x, float step, int nbit, bool sign) {
	int pos_end = sign ?  (1 << (nbit - 1)) - 1 : (1 << nbit) - 1;
	int neg_end = sign ? -(1 << (nbit - 1))     : 0;
	int raw_q = roundf(x/step);
	return raw_q > pos_end ? pos_end : (raw_q < neg_end ? neg_end : raw_q);
}

// float forward(float* x, float* y, int** layers, float* weights, float* biases, float* wq_steps, float* xq_steps) {

// }

void run_sim_fast_approx_ma() {
	int layers[8][5] = {
		{ 1, 64, 3, 1, 1},
		{64, 32, 1, 1, 0},
		{32, 32, 3, 1, 1},
		{32, 32, 3, 1, 1},
		{32, 32, 3, 1, 1},
		{32, 32, 3, 1, 1},
		{32, 64, 1, 1, 0},
		{64,  4, 3, 1, 1},
	};

	float wq_steps[8] = {1.0/(1<<10), 1.0/(1<<8), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<8), 0.0};
	float xq_steps[8] = {1.0/(1<< 8), 1.0/(1<<8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<<8), 0.0};

	float** weights = new float*[8];
	float** biases = new float*[8];

	for (int i = 0; i < 1; ++i) {
		std::string data_file_name = "./data/layer_" + std::to_string(i);
		FILE* f = fopen(data_file_name.c_str(), "r");

		int c = layers[i][0];
		int n = layers[i][1];
		int k = layers[i][2];
		int s = layers[i][3];
		int p = layers[i][4];

		int weight_size = c * n * k * k;
		weights[i] = new float[weight_size];

		int bias_size = n;
		biases[i] = new float[bias_size];

		int buffer_size = weight_size + bias_size;
		float* buffer = new float[buffer_size];
		fread(buffer, sizeof(float), buffer_size, f);

		copy_cpu(bias_size, buffer, 1, biases[i], 1);
		copy_cpu(weight_size, buffer + bias_size, 1, weights[i], 1);
		
		for (int j = 0; j < 10; ++j) {
			std::cout << biases[i][j] << std::endl;
		}
		std::cout << std::endl;
		for (int j = 0; j < 10; ++j) {
			std::cout << weights[i][j] << std::endl;
		}
		delete buffer;
	}


	for (int i = 0; i < 8; ++i) {
		delete[] weights[i];
		delete[] biases[i];
	}
	delete[] weights;
	delete[] biases;
}

int main() {
	run_sim_fast_approx_ma();
	return 0;
}