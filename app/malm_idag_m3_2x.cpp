#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "ssdn.h"

static int N_IMAGE;
static int ECC_ITER;
static std::string MODEL_PATH;
static std::string DATA_DIR;
static float WP;
static void (*APPROX_FUNC)(float*, float*, int, int, int, int, int, float*, int)  = distribute_approximate_iterative_gpu;

static void print_array(float* x, int n, int m, int k) {
	float* _cpu =  new float[n*m*k];
	cuda_pull_array(x, _cpu, n * m * k);

	for (int ni = 0; ni < n; ++ni) {
		for (int mi = 0; mi < m; ++mi) {
			for (int ki = 0; ki < k; ++ki) {
				std::cout << _cpu[ni*m*k + mi*k + ki] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	delete[] _cpu;
}

static float* conv2d_memsafe(float* x, int xw, int xh,
	float* w, float* b, int* desc,
	float xq_step, float wq_step,
	int& yw, int& yh, int& yn) {

	int c = desc[0];
	int n = desc[1];
	int k = desc[2];
	int s = desc[3];
	int p = desc[4];

	if (xq_step > 0 && ECC_ITER > 0) {
		quantize_gpu(xw * xh * c, xq_step, 10, false, x);
	}

	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;
	
	float* x_padded = cuda_make_array(0, xpw * xph * c);
	padding_gpu(x, xw, xh, c, p, x_padded);
	cuda_free(x); // x expires

	float* x_mat = cuda_make_array(0, xpw * xph * c * k * k);
	unrolling_gpu(x_padded, xpw, xph, c, k, s, x_mat);
	cuda_free(x_padded); // x_padded expires

	yw = (xpw - k) / s + 1;
	yh = (xph - k) / s + 1;
	yn = n;

	int y_size = yw * yh;
	int f_size = k * k * c;

	float* y = cuda_make_array(0, y_size * n);
	tile_repeat_gpu(n, 1, y_size, b, 1, y, 1);

	float* xw_mat = cuda_make_array(0, y_size * f_size * 1);
	for (int ni = 0; ni < n; ++ni) {
		if (xq_step > 0 && ECC_ITER > 0) {
			// distribute_mul_gpu(x_mat, w + ni * f_size, yw, yh, c, k, 1, xw_mat);
			APPROX_FUNC(x_mat, w + ni * f_size, yw, yh, c, k, 1, xw_mat, ECC_ITER);
			scale_gpu(y_size * f_size * 1, xq_step * wq_step, xw_mat, 1);
		} else {
			distribute_mul_gpu(x_mat, w + ni * f_size, yw, yh, c, k, 1, xw_mat);
		}
		accumulate_gpu(y_size, f_size, xw_mat, 1, y + ni * y_size, 1);
	}

	cuda_free(x_mat); // x_mat expires
	cuda_free(xw_mat); // xw_mat expires
	return y;
}

static float* conv2d(float* x, int xw, int xh,
	float* w, float* b, int* desc,
	float xq_step, float wq_step,
	int& yw, int& yh, int& yn) {

	int c = desc[0];
	int n = desc[1];
	int k = desc[2];
	int s = desc[3];
	int p = desc[4];

	if (xq_step > 0 && ECC_ITER > 0) {
		quantize_gpu(xw * xh * c, xq_step, 10, false, x);
	}

	int xpw = xw + 2 * p;
	int xph = xh + 2 * p;
	
	float* x_padded = cuda_make_array(0, xpw * xph * c);
	padding_gpu(x, xw, xh, c, p, x_padded);
	cuda_free(x); // x expires

	float* x_mat = cuda_make_array(0, xpw * xph * c * k * k);
	unrolling_gpu(x_padded, xpw, xph, c, k, s, x_mat);
	cuda_free(x_padded); // x_padded expires

	yw = (xpw - k) / s + 1;
	yh = (xph - k) / s + 1;
	yn = n;

	int y_size = yw * yh;
	int f_size = k * k * c;

	float* xw_mat = cuda_make_array(0, y_size * f_size * n);
	if (xq_step > 0 && ECC_ITER > 0) {
		// distribute_mul_gpu(x_mat, w, yw, yh, c, k, n, xw_mat);
		APPROX_FUNC(x_mat, w, yw, yh, c, k, n, xw_mat, ECC_ITER);
		scale_gpu(y_size * f_size * n, xq_step * wq_step, xw_mat, 1);
	} else {
		distribute_mul_gpu(x_mat, w, yw, yh, c, k, n, xw_mat);
	}
	cuda_free(x_mat); // x_mat expires

	float* y = cuda_make_array(0, y_size * n);
	tile_repeat_gpu(n, 1, y_size, b, 1, y, 1);
	accumulate_gpu(y_size * n, f_size, xw_mat, 1, y, 1);
	cuda_free(xw_mat); // xw_mat expires
	return y;
}

static float forward(float* im, int imw, int imh,
	float* gt, int gtw, int gth,
	int** layers, int n_layer, float** weights, float** biases,
	float* wq_steps, float* xq_steps,
	float*& z_im_c
	) {

	int zw, zh, zn;
	float* z = cuda_make_array(0, imw * imh);
	copy_gpu(imw * imh, im, 1, z, 1);

	float* x = z;
	float* y;
	int xw = imw;
	int xh = imh;

	for (int li = 0; li < n_layer; ++li) {
		if (xw * xh > 400 * 400) {
			y = conv2d_memsafe(x, xw, xh, weights[li], biases[li], layers[li], xq_steps[li], wq_steps[li], zw, zh, zn);
		} else {
			y = conv2d(x, xw, xh, weights[li], biases[li], layers[li], xq_steps[li], wq_steps[li], zw, zh, zn);
		}
		if (li != n_layer-1) {
			min_gpu(zw*zh*zn, 0, y, 1);
		}
		z = y;
		x = z;
		xw = zw;
		xh = zh;
	}

	float* z_im = cuda_make_array(0, zw*zh*zn);
	flatten_arrange_gpu(im, z, zw, zh, 2, z_im);
	z_im_c = new float[gtw * gth + 2];
	z_im_c[0] = (float)gtw;
	z_im_c[1] = (float)gth;
	cuda_pull_array(z_im, z_im_c + 2, gtw * gth);


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

	cuda_free(z);
	cuda_free(t);
	cuda_free(z_im);

	return -10 * log10(mean);
}

static int parse(int argc, char** argv) {
	MODEL_PATH = argv[2];
	
	if (0 == strcmp(argv[3], "set14b")) {
		N_IMAGE = 14;
		DATA_DIR = "data/set14b/";
	} else if (0 == strcmp(argv[3], "set5b")) {
		N_IMAGE = 5;
		DATA_DIR = "data/set5b/";
	} else if (0 == strcmp(argv[3], "b100b")) {
		N_IMAGE = 100;
		DATA_DIR = "data/b100b/";
	} else if (0 == strcmp(argv[3], "urban100b")) {
		N_IMAGE = 100;
		DATA_DIR = "data/urban100b/";
	} else {
		std::cout << "[ERRO] unrecognized dataset: " << argv[3] << std::endl;
		return 1;
	}
	std::cout << "[INFO] using dataset: " << DATA_DIR << std::endl;
	
	if (0 == strcmp(argv[4], "0")) {
		ECC_ITER = 0;
		std::cout << "[INFO] turn off quantization" << std::endl;
	} else if (0 == strcmp(argv[4], "1")) {
		ECC_ITER = 1;
		std::cout << "[INFO] setting ECC_ITER = " << ECC_ITER << std::endl;
	} else if (0 == strcmp(argv[4], "2")) {
		ECC_ITER = 2;
		std::cout << "[INFO] setting ECC_ITER = " << ECC_ITER << std::endl;
	} else if (0 == strcmp(argv[4], "3")) {
		ECC_ITER = 3;
		std::cout << "[INFO] setting ECC_ITER = " << ECC_ITER << std::endl;
	} else if (0 == strcmp(argv[4], "4")) {
		ECC_ITER = 4;
		std::cout << "[INFO] setting ECC_ITER = " << ECC_ITER << std::endl;
	}

	if (0 == strcmp(argv[5], "--ilm")) {
		std::cout << "[INFO] using ILM approximation" << std::endl;
		APPROX_FUNC = distribute_approximate_ilm_iterative_gpu;
	} else if (0 == strcmp(argv[5], "--ori")) {
		std::cout << "[INFO] using original approximation" << std::endl;
		APPROX_FUNC = distribute_approximate_one_side_iterative_gpu;
	} else if (0 == strcmp(argv[5], "--signedap")) {
		std::cout << "[INFO] using signedap approximation" << std::endl;
		APPROX_FUNC = distribute_approximate_signedap_iterative_gpu;
	} else {
		std::cout << "[INFO] using custom (min-error) approximation" << std::endl;
	}
	
	return 0;
}

int malm_idag_m3_2x(int argc, char** argv) {
	if (parse(argc, argv)) {
		std::cout << "[ERRO] parsing error" << std::endl;
		return 1;
	}
	
	WP = 0.011;
	std::cout << "[INFO] WP = " << WP << std::endl;

	// m3
	int _layers[40] = { 1, 64, 3, 1, 1, 64, 32, 1, 1, 0, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 64, 1, 1, 0, 64,  4, 3, 1, 1};

	// m1
	// int _layers[40] = { 1, 32, 3, 1, 1, 32, 16, 1, 1, 0, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 32, 1, 1, 0, 32,  4, 3, 1, 1};

	// m2
	// int _layers[40] = { 1, 48, 3, 1, 1, 48, 24, 1, 1, 0, 24, 24, 3, 1, 1, 24, 24, 3, 1, 1, 24, 24, 3, 1, 1, 24, 24, 3, 1, 1, 24, 48, 1, 1, 0, 48,  4, 3, 1, 1};

	int **layers = new int*[8];
	for (int i = 0; i < 8; ++i) {
		layers[i] = new int[5];
	}
	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 5; ++j) {
			layers[i][j] = _layers[i*5+j];
		}
	}

	// m3
	// float wq_steps[8] = {1.0/(1<<9), 1.0/(1<<7), 1.0/(1<<9), 1.0/(1<<9), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<11), 0.0};
	// float wq_steps[8] = {1.0/(1<<10), 1.0/(1<<8), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<8), 0.0};
	float wq_steps[8] = {1.0/(1<<9), 1.0/(1<<7), 1.0/(1<<9), 1.0/(1<<9), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<11), 1.0/(1<<12)};
	
	// float xq_steps[8] = {1.0/(1<< 8), 1.0/(1<<8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<<8), 0.0};
	float xq_steps[8] = {1.0/(1<< 8), 1.0/(1<<8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<<8), 1.0/(1<<8)};

	// m1
	// float wq_steps[8] = {1.0/(1<< 8), 1.0/(1<<9), 1.0/(1<<10), 1.0/(1<< 7), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 0.0};
	// float xq_steps[8] = {1.0/(1<< 8), 1.0/(1<<8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 0.0};

	// m2
	// float wq_steps[8] = {1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 1.0/(1<<10), 0.0};
	// float xq_steps[8] = {1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 1.0/(1<< 8), 0.0};

	float** weights = new float*[8];
	float** biases = new float*[8];

	// load model
	for (int i = 0; i < 8; ++i) {
		std::cout << "[INFO] load layer " << i << std::endl;
		// std::string data_file_name = MODEL_PATH + "/layer_" + std::to_string(i);
		std::string data_file_name = MODEL_PATH + "/" + std::to_string(i);
		FILE* f = fopen(data_file_name.c_str(), "r");
		if (!f) {
			std::cout << "[ERRO] cannot load model from " + MODEL_PATH << std::endl;
			return 1;
		}

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
		if (wq_steps[i] > 0 && ECC_ITER > 0) {
			// quantize_gpu(weight_size, wq_steps[i], 11, true, weights[i]);
			// compensate_WP_gpu(weight_size, WP, weights[i]);
			// max_gpu(weight_size, (1 << 10) - 1, weights[i]);
			// min_gpu(weight_size, -1 << 10, weights[i]);

			quantize_compensate_wp_gpu(weight_size, wq_steps[i], 11, WP, true, weights[i]);
			print_array(weights[i], 1, 1, k*k);

			// quantize_gpu(weight_size, wq_steps[i], 11, true, weights[i]);
			// compensate_log_gpu(weight_size, WP, 1 << 10, weights[i]);
			// max_gpu(weight_size, (1 << 10) - 1, weights[i]);
			// min_gpu(weight_size, -1 << 10, weights[i]);
		}

		delete[] _weight;

		fclose(f);
	}

	// load images
	float psnr_mean = 0;

	mkdir((DATA_DIR + "_" + std::to_string(ECC_ITER) + "_" + std::to_string(WP)).c_str(), 0777);
	for (int i = 0; i < N_IMAGE; ++i) {
		std::string data_file_name;
		FILE* f;
		float wf, hf;
		
		// read input image
		data_file_name = DATA_DIR + "/im_" + std::to_string(i);
		f = fopen(data_file_name.c_str(), "r");
		
		fread(&wf, sizeof(float), 1, f);
		fread(&hf, sizeof(float), 1, f);
		int imw = int(wf);
		int imh = int(hf);

		int im_size = imw * imh;
		float* _im = new float[im_size];
		fread(_im, sizeof(float), im_size, f);
		float* im = cuda_make_array(_im, im_size);
		delete[] _im;

		fclose(f);

		// read ground truth image
		data_file_name = DATA_DIR + "/gt_" + std::to_string(i);
		f = fopen(data_file_name.c_str(), "r");
		
		fread(&wf, sizeof(float), 1, f);
		fread(&hf, sizeof(float), 1, f);
		int gtw = int(wf);
		int gth = int(hf);

		int gt_size = gtw * gth;
		float* _gt = new float[gt_size];
		fread(_gt, sizeof(float), gt_size, f);
		float* gt = cuda_make_array(_gt, gt_size);
		delete[] _gt;

		fclose(f);

		// forwarding
		float* output_c;
		float psnr = forward(im, imw, imh, gt, gtw, gth, layers, 8, weights, biases, wq_steps, xq_steps, output_c);

		f = fopen((DATA_DIR + "_" + std::to_string(ECC_ITER) + "_" + std::to_string(WP) + "/ou_" + std::to_string(i)).c_str(), "wb");
		fwrite(output_c, sizeof(float), gtw * gth + 2, f);
		fclose(f);
		delete[] output_c;

		psnr_mean += psnr;
		std::cout << psnr << std::endl;
		cuda_free(im);
		cuda_free(gt);
	}
	std::cout << psnr_mean / N_IMAGE << std::endl;

	for (int i = 0; i < 8; ++i) {
		cuda_free(weights[i]);
		cuda_free(biases[i]);
	}
	for (int i = 0; i < 8; ++i) {
		delete[] layers[i];
	}
	delete[] layers;
	delete[] weights;
	delete[] biases;
	return 0;
}