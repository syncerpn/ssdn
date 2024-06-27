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

float* padding(float* x, int xw, int xh, int p) {
	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;
	float* xp = new float[xpw * xph];
	fill_cpu(xpw * xph, 0, xp, 1);
	for (int i = 1; i < xph - 1; ++i) {
		copy_cpu(xw, x+(i-1)*xw, 1, xp+i*xpw+p, 1);
	}
	return xp;
}

float* unroll(float* x, int xw, int xh, int c, int k, int s) {
	int yw = (xw - k) / s + 1;
	int yh = (xh - k) / s + 1;

	int y_size = yw * yh;
	int f_size = k * k * c;

	float* x_mat = new float[yw * yh * f_size];

	for (int hi = 0; hi < yh; ++hi) {
		for (int wi = 0; wi < yw; ++wi) {
			for (int ci = 0; ci < c; ++ci) {
				for (int ki = 0; ki < k; ++ki) {
					copy_cpu(k, x+ci*xh*xw+(hi*s+ki)*xw+wi*s, 1, x_mat+(hi*yw+wi)*k*k*c+ci*k*k+ki*k, 1);
				}
			}
		}
	}
	return x_mat;
}

void conv2d(float* x, int xw, int xh, float* w, float* b, int* desc, float xq_step, float wq_step) {
	int c = desc[0];
	int n = desc[1];
	int k = desc[2];
	int s = desc[3];
	int p = desc[4];

	float* x_padded = padding(x, xw, xh, p);
	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;

	float* x_mat = unroll(x_padded, xpw, xph, c, k, s);

	delete[] x_padded;
	delete[] x_mat;
}

float forward(float* x, int xw, int xh,
	float* y, int yw, int yh,
	int** layers, float* weights, float* biases,
	float* wq_steps, float* xq_steps) {



	return 0;
}

void run_sim_fast_approx_ma() {
	int _layers[40] = { 1, 64, 3, 1, 1, 64, 32, 1, 1, 0, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 64, 1, 1, 0, 64,  4, 3, 1, 1};
	int **layers = new int*[8];
	for (int i = 0; i < 8; ++i) {
		layers[i] = new int[5];
	}
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 5; ++j) {
			layers[i][j] = _layers[i*5+j];
		}
	}

	float wq_steps[8] = {1.0/(1<<10), 1.0/(1<<8), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<8), 0.0};
	float xq_steps[8] = {1.0/(1<< 8), 1.0/(1<<8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<<8), 0.0};

	float** weights = new float*[8];
	float** biases = new float*[8];

	// load model
	for (int i = 0; i < 8; ++i) {
		std::string data_file_name = "./data/layer_" + std::to_string(i);
		FILE* f = fopen(data_file_name.c_str(), "r");

		int c = layers[i][0];
		int n = layers[i][1];
		int k = layers[i][2];
		int s = layers[i][3];
		int p = layers[i][4];

		int bias_size = n;
		biases[i] = new float[bias_size];
		fread(biases[i], sizeof(float), bias_size, f);

		int weight_size = c * n * k * k;
		weights[i] = new float[weight_size];
		fread(weights[i], sizeof(float), weight_size, f);

		fclose(f);
	}

	// load images
	for (int i = 0; i < 1; ++i) {
		std::string data_file_name;
		FILE* f;
		float wf, hf;
		
		data_file_name = "./data/imd_" + std::to_string(i);
		f = fopen(data_file_name.c_str(), "r");
		
		fread(&wf, sizeof(float), 1, f);
		fread(&hf, sizeof(float), 1, f);
		int imw = int(wf);
		int imh = int(hf);

		int im_size = imw * imh;
		float* im = new float[im_size];
		fread(im, sizeof(float), im_size, f);

		fclose(f);

		data_file_name = "./data/gtd_" + std::to_string(i);
		f = fopen(data_file_name.c_str(), "r");
		
		fread(&wf, sizeof(float), 1, f);
		fread(&hf, sizeof(float), 1, f);
		int gtw = int(wf);
		int gth = int(hf);

		int gt_size = gtw * gth;
		float* gt = new float[gt_size];
		fread(gt, sizeof(float), gt_size, f);

		fclose(f);

		float psnr = forward(im, imw, imh, gt, gtw, gth, layers, weights, biases, wq_steps, xq_steps);

		delete[] im;
		delete[] gt;
	}

	for (int i = 0; i < 8; ++i) {
		delete[] weights[i];
		delete[] biases[i];
	}
	delete[] weights;
	delete[] biases;
}

int main() {
	// run_sim_fast_approx_ma();
	int xw = 5;
	int xh = 4;
	int c = 1;
	int k = 3;
	int p = 1;
	int s = 1
	float* x = new float[xw * xh * c];
	for (int ic = 0; ic < c; ++ic) {
		for (int ih = 0; ih < xh; ++ih) {
			for (int iw = 0; iw < xw; ++iw) {
				x[ic*xh*xw+ih*xw+iw] = ic*xh*xw+ih*xw+iw;
				std::cout << x[ic*xh*xw+ih*xw+iw] << " ";
			}
			std::cout << std::endl;
		}
	}
	float* x_padded = padding(x, xw, xh, p);
	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;

	for (int ic = 0; ic < c; ++ic) {
		for (int ih = 0; ih < xph; ++ih) {
			for (int iw = 0; iw < xpw; ++iw) {
				std::cout << x_padded[ic*xph*xpw+ih*xpw+iw] << " ";
			}
			std::cout << std::endl;
		}
	}

	float* x_mat = unroll(x_padded, xpw, xph, c, k, s);

	int yw = (xpw - k) / s + 1;
	int yh = (xph - k) / s + 1;

	int y_size = yw * yh;
	int f_size = k * k * c;

	for (int hi = 0; hi < yh; ++hi) {
		for (int wi = 0; wi < yw; ++wi) {
			for (int kki = 0; kki < c*k*k; ++kki) {
				std::cout << x_mat[(hi*yw+wi)*k*k*c + kki] << " ";
			}
			std::cout << std::endl;
		}
	}

	delete[] x;
	delete[] x_padded;
	delete[] x_mat;
	return 0;
}