#include <iostream>
#include <fstream>
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

	for (int i = 0; i < 1; ++i) {
		std::string data_file_name = "./data/layer_" + std::to_string(i);
		std::ifstream df(data_file_name);
		if (!df) {
			std::cout << "[ERRO] failed to open file" << std::endl;
			continue;
		}
		std::vector<float> ft;
		std::size_t n = df.tellg() / sizeof(float);
		std::vector<float> buffer(n);

        if (!df.read(reinterpret_cast<char*>(buffer.data()), n)) {
            continue; // Skip to the next file if there's an error
        }

        // Append the buffer to the allFloatValues vector
        ft.insert(ft.end(), buffer.begin(), buffer.end());

		df.close();
	}
}

int main() {
	run_sim_fast_approx_ma();
	return 0;
}