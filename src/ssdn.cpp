#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "ssdn.h"

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
	float* xw_mat = workspace[2];
	float* y = workspace[3];

	if (xq_step > 0) {
		quantize_gpu(xw * xh * c, xq_step, 10, false, x);
	}

	padding_gpu(x, xw, xh, c, p, x_padded);
	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;

	unrolling_gpu(x_padded, xpw, xph, c, k, s, x_mat);
	yw = (xpw - k) / s + 1;
	yh = (xph - k) / s + 1;
	yn = n;

	int y_size = yw * yh;
	int f_size = k * k * c;

	if (xq_step > 0) {
		distribute_approximate_gpu(x_mat, w, yw, yh, c, k, n, xw_mat);
		scale_gpu(y_size * f_size * n, xq_step * wq_step, xw_mat, 1);
	} else {
		distribute_mul_gpu(x_mat, w, yw, yh, c, k, n, xw_mat);
	}

	tile_repeat_gpu(n, 1, y_size, b, 1, y, 1);
	accumulate_gpu(y_size * n, f_size, xw_mat, 1, y, 1);
}

float forward(float* im, int imw, int imh,
	float* gt, int gtw, int gth,
	int** layers, int n_layer, float** weights, float** biases,
	float* wq_steps, float* xq_steps, float** workspace) {

	int zw, zh, zn;
	float* z = workspace[3];

	copy_gpu(imw * imh, im, 1, z, 1);

	float* x = z;
	int xw = imw;
	int xh = imh;

	for (int li = 0; li < n_layer; ++li) {
		// std::cout << "[INFO] layer " << li;
		conv2d(x, xw, xh, weights[li], biases[li], layers[li], xq_steps[li], wq_steps[li], zw, zh, zn, workspace);
		if (li != n_layer-1) {
			min_gpu(zw*zh*zn, 0, z, 1);
		}
		x = z;
		xw = zw;
		xh = zh;
		// std::cout << " done" << std::endl;
	}

	float* z_im = cuda_make_array(0, zw*zh*zn);
	flatten_arrange_gpu(im, z, zw, zh, 2, z_im);

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
	cuda_free(z_im);

	return -10 * log10(mean);
}

void run_sim_fast_approx_ma(float wp) {
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

	const size_t SPA_SIZE_MAX = 103680;
	const size_t N_MAX = 64;
	const size_t C_MAX = 32;
	const size_t K_MAX = 3;

	float** weights = new float*[8];
	float** biases = new float*[8];
	float** workspace = new float*[4];
	std::cout << "[INFO] allocating GPU mem for processing" << std::endl;
	workspace[0] = cuda_make_array(0, SPA_SIZE_MAX * N_MAX); // x_padded
	workspace[1] = cuda_make_array(0, SPA_SIZE_MAX * N_MAX * K_MAX * K_MAX); // x_mat
	workspace[2] = cuda_make_array(0, SPA_SIZE_MAX * N_MAX * K_MAX * K_MAX * C_MAX); // xw_mat
	workspace[3] = cuda_make_array(0, SPA_SIZE_MAX * N_MAX); // y
	std::cout << "[INFO] allocation finished" << std::endl;
	// load model
	for (int i = 0; i < 8; ++i) {
		std::string data_file_name = "./data/m3_33.113/layer_" + std::to_string(i);
		FILE* f = fopen(data_file_name.c_str(), "r");

		int c = layers[i][0];
		int n = layers[i][1];
		int k = layers[i][2];

		int bias_size = n;
		float* _bias = new float[bias_size];
		fread(_bias, sizeof(float), bias_size, f);
		biases[i] = cuda_make_array(_bias, bias_size);
		delete[] _bias;

		int weight_size = c * n * k * k;
		float* _weight = new float[weight_size];
		fread(_weight, sizeof(float), weight_size, f);
		weights[i] = cuda_make_array(_weight, weight_size);

		// add quantization
		if (wq_steps[i] > 0) {
			// quantize_gpu(weight_size, wq_steps[i], 11, true, weights[i]);
			// compensate_wp_gpu(weight_size, wp, weights[i]);
			// max_gpu(weight_size, (1 << 10) - 1, weights[i]);
			// min_gpu(weight_size, -1 << 10, weights[i]);

			// quantize_compensate_wp_gpu(weight_size, wq_steps[i], 11, true, weights[i]);

			quantize_gpu(weight_size, wq_steps[i], 11, true, weights[i]);
			compensate_log_gpu(weight_size, wp, 1 << 10, weights[i]);
			max_gpu(weight_size, (1 << 10) - 1, weights[i]);
			min_gpu(weight_size, -1 << 10, weights[i]);
		}

		delete[] _weight;

		fclose(f);
	}

	float* im = cuda_make_array(0, SPA_SIZE_MAX);
	float* gt = cuda_make_array(0, SPA_SIZE_MAX * 4);
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

	for (int i = 0; i < 8; ++i) {
		cuda_free(weights[i]);
		cuda_free(biases[i]);
	}
	for (int i = 0; i < 8; ++i) {
		delete[] layers[i];
	}
	delete[] layers;
	delete[] workspace;
	delete[] weights;
	delete[] biases;
}

int main() {
	for (int i = 0; i < 1000; ++i) {
		float wp = i / 1000.0;
		std::cout << "[INFO] wp = " << wp << std::endl;
		run_sim_fast_approx_ma(wp);
	}
}