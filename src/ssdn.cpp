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

void padding(float* x, int xw, int xh, int c, int p, float* xp) {
	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;
	for (int ci = 0; ci < c; ++ci) {
		for (int i = p; i < xph - p; ++i) {
			copy_gpu(xw, x+ci*xh*xw+(i-p)*xw, 1, xp+ci*xph*xpw+i*xpw+p, 1);
		}
	}
}

void unroll(float* x, int xw, int xh, int c, int k, int s, float* x_mat) {
	int yw = (xw - k) / s + 1;
	int yh = (xh - k) / s + 1;

	int y_size = yw * yh;
	int f_size = k * k * c;

	for (int hi = 0; hi < yh; ++hi) {
		for (int wi = 0; wi < yw; ++wi) {
			for (int ci = 0; ci < c; ++ci) {
				for (int ki = 0; ki < k; ++ki) {
					copy_gpu(k, x+ci*xh*xw+(hi*s+ki)*xw+wi*s, 1, x_mat+(hi*yw+wi)*f_size+ci*k*k+ki*k, 1);
				}
			}
		}
	}
}

void conv2d(float* x, int xw, int xh,
	float* w, float* b, int* desc,
	float xq_step, float wq_step,
	int& yw, int& yh, int& yn,
	float** workspace) {

	int c = desc[0];
	int n = desc[1];
	int k = desc[2];
	int s = desc[3];
	int p = desc[4];

	float* x_padded = workspace[0];
	float* x_mat = workspace[1];
	float* x_mat_r = workspace[2];
	float* w_mat_r = workspace[3];
	float* b_mat_r = workspace[4];
	float* y = workspace[5];

	padding(x, xw, xh, c, p, x_padded);
	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;

	unroll(x_padded, xpw, xph, c, k, s, x_mat);
	yw = (xpw - k) / s + 1;
	yh = (xph - k) / s + 1;
	yn = n;

	int y_size = yw * yh;
	int f_size = k * k * c;

	tile_repeat_gpu(f_size * y_size, f_size * y_size, n, x_mat, 1, x_mat_r, 1);
	tile_repeat_gpu(f_size * n, f_size, y_size, w, 1, w_mat_r, 1);
	tile_repeat_gpu(n, 1, y_size, b, 1, b_mat_r, 1);
	mul_gpu(f_size * n * y_size, w_mat_r, 1, x_mat_r, 1);
	accumulate_gpu(y_size * n, f_size, x_mat_r, 1, y, 1);
	axpy_gpu(y_size * n, 1, b_mat_r, 1, y, 1);
}

float forward(float* im, int imw, int imh,
	float* gt, int gtw, int gth,
	int** layers, int n_layer, float** weights, float** biases,
	float* wq_steps, float* xq_steps, float** workspace) {

	float* x = im;
	int xw = imw;
	int xh = imh;

	int zw, zh, zn;
	float* z = workspace[5];
	for (int li = 0; li < n_layer; ++li) {
		std::cout << "[INFO] layer " << li;
		conv2d(x, xw, xh, weights[li], biases[li], layers[li], xq_steps[li], wq_steps[li], zw, zh, zn, workspace);
		if (li != n_layer-1) {
			min_gpu(zw*zh*zn, 0, z, 1);
		}
		x = z;
		xw = zw;
		xh = zh;
		std::cout << " done" << std::endl;
	}

	float* z_im = new float[zw*zh*zn]();
	for (int ni = 0; ni < 4; ++ni) {
		int nhi = ni / 2;
		int nwi = ni % 2;
		for (int hi = 0; hi < zh; ++hi) {
			for (int wi = 0; wi < zw; ++wi) {
				z_im[(hi*2+nhi)*zw*2+wi*2+nwi] = im[hi*zw+wi] + z[ni*zh*zw+hi*zw+wi];
			}
		}
	}

	axpy_gpu(gtw*gth, -1, gt, 1, z_im, 1);
	pow_gpu(gtw*gth, 2, z_im, 1);
	float sum = 0;
	float* t = cuda_make_array(0, 1);
	float _t;
	for (int hi = 2; hi < gth - 2; ++hi) {
		fill_gpu(1, 0, t, 1);
		accumulate_gpu(1, gtw-4, z_im+hi*gtw+2, 1, t, 1);
		cuda_pull_array(t, &_t, 1);
		sum += _t;
	}
	float mean = sum / ((gth-4)*(gtw-4));

	cuda_free(t);
	cuda_free(z);
	cuda_free(z_im);

	return -10 * log10(mean);
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

	const size_t H_MAX = 256;
	const size_t W_MAX = 256;
	const size_t N_MAX = 64;
	const size_t C_MAX = 32;
	const size_t K_MAX = 3;

	float** weights = new float*[8];
	float** biases = new float*[8];
	float** workspace = new float*[6];
	workspace[0] = cuda_make_array(0, H_MAX * W_MAX * N_MAX); // x_padded
	workspace[1] = cuda_make_array(0, H_MAX * W_MAX * N_MAX * K_MAX * K_MAX); // x_mat
	workspace[2] = cuda_make_array(0, H_MAX * W_MAX * N_MAX * K_MAX * K_MAX * C_MAX); // x_mat_r
	workspace[3] = cuda_make_array(0, H_MAX * W_MAX * N_MAX * K_MAX * K_MAX * C_MAX); // w_mat_r
	workspace[4] = cuda_make_array(0, H_MAX * W_MAX * N_MAX); // b_mat_r
	workspace[5] = cuda_make_array(0, H_MAX * W_MAX * N_MAX); // y

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
		float* _bias = new float[bias_size];
		fread(_bias, sizeof(float), bias_size, f);
		biases[i] = cuda_make_array(_bias, bias_size);
		delete[] _bias;

		int weight_size = c * n * k * k;
		float* _weight = new float[weight_size];
		fread(_weight, sizeof(float), weight_size, f);
		weights[i] = cuda_make_array(_weight, weight_size);
		delete[] _weight;

		fclose(f);
	}

	float* im = cuda_make_array(0, H_MAX * W_MAX);
	float* gt = cuda_make_array(0, H_MAX * W_MAX * 4);
	// load images
	float psnr_mean = 0;
	for (int i = 0; i < 14; ++i) {
		std::string data_file_name;
		FILE* f;
		float wf, hf;
		
		// read input image
		data_file_name = "./data/imd_" + std::to_string(i);
		f = fopen(data_file_name.c_str(), "r");
		
		fread(&wf, sizeof(float), 1, f);
		fread(&hf, sizeof(float), 1, f);
		int imw = int(wf);
		int imh = int(hf);

		int im_size = imw * imh;
		float* _im = new float[im_size];
		fread(_im, sizeof(float), im_size, f);
		cuda_push_array(im, _im, im_size);
		delete[] _im;

		fclose(f);

		// read ground truth image
		data_file_name = "./data/gtd_" + std::to_string(i);
		f = fopen(data_file_name.c_str(), "r");
		
		fread(&wf, sizeof(float), 1, f);
		fread(&hf, sizeof(float), 1, f);
		int gtw = int(wf);
		int gth = int(hf);

		int gt_size = gtw * gth;
		float* _gt = new float[gt_size];
		fread(_gt, sizeof(float), gt_size, f);
		cuda_push_array(gt, _gt, gt_size);
		delete[] _gt;

		fclose(f);

		// forwarding
		float psnr = forward(im, imw, imh, gt, gtw, gth, layers, 8, weights, biases, wq_steps, xq_steps, workspace);
		psnr_mean += psnr;
		std::cout << psnr << std::endl;

	}
	std::cout << psnr_mean / 14 << std::endl;

	cuda_free(im);
	cuda_free(gt);
	cuda_free(workspace[0]);
	cuda_free(workspace[1]);
	cuda_free(workspace[2]);
	cuda_free(workspace[3]);
	cuda_free(workspace[4]);
	cuda_free(workspace[5]);

	for (int i = 0; i < 8; ++i) {
		cuda_free(weights[i]);
		cuda_free(biases[i]);
	}
	delete[] workspace;
	delete[] weights;
	delete[] biases;
}

int main() {
	run_sim_fast_approx_ma();
	// int xw = 5;
	// int xh = 4;
	// int c = 2;
	// int k = 3;
	// int p = 1;
	// int s = 1;
	// int n = 4;

	// float* x = new float[xw * xh * c];
	// for (int ic = 0; ic < c; ++ic) {
	// 	for (int ih = 0; ih < xh; ++ih) {
	// 		for (int iw = 0; iw < xw; ++iw) {
	// 			x[ic*xh*xw+ih*xw+iw] = ic*xh*xw+ih*xw+iw;
	// 			std::cout << x[ic*xh*xw+ih*xw+iw] << " ";
	// 		}
	// 		std::cout << std::endl;
	// 	}
	// }
	// std::cout << std::endl;

	// float* w = new float[k*k*c*n];
	// for (int i = 0; i < k*k*c*n; ++i) {
	// 	w[i] = (float)i / 2;
	// 	std::cout << w[i] << " ";
	// }
	// std::cout << std::endl << std::endl;

	// float* b = new float[n];
	// for (int i = 0; i < n; ++i) {
	// 	b[i] = (float)i / 3;
	// 	std::cout << b[i] << " ";
	// }

	// int yw, yh, yn;
	// int ldesc[5] = {c, n, k, s, p};
	// float* y = conv2d(x, xw, xh, w, b, ldesc, 0, 0, yw, yh, yn);

	// for (int ni = 0; ni < n; ++ni) {
	// 	for (int hi = 0; hi < yh; ++hi) {
	// 		for (int wi = 0; wi < yw; ++wi) {
	// 			std::cout << y[ni*yh*yw+hi*yw+wi] << " ";
	// 		}
	// 		std::cout << std::endl;
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;

	// delete[] x;
	// delete[] w;
	// delete[] y;
	// delete[] b;
	return 0;
}