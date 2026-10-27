#include <iostream>
#include <math.h>
#include <sys/stat.h>

#include <vector>
#include <unordered_map>
#include <string>

#include "ssdn.h"

#include <chrono>
#define TIME_START start = std::chrono::high_resolution_clock::now();
#define TIME_STOP(prefix) std::cout << prefix << " :" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << " microseconds" << std::endl;

static int N_IMAGE;
static int ECC_ITER;
static std::string MODEL_PATH;
static std::string DATA_DIR;
static float WP = 0.0;
static void (*APPROX_FUNC)(float*, float*, int, int, int, int, float*, int)  = conv2d_chunk_mul_gpu;

static std::unordered_map<std::string, float*> node;
static size_t node_size = 268435456;
static std::vector<std::string> node_names = {"x", "y", "z", "x_head"};


static float* workspace; // continuous buffer with size ws_size and ws_delta as separation
static size_t ws_size = 268435456;
static size_t ws_delta;

static float* request_workspace(size_t s, bool empty) {
	if (ws_delta + s > ws_size) {
		ws_delta = 0;
	}
	float* p = workspace + ws_delta;
	ws_delta += s;
	if (empty) {
		fill_gpu(s, 0, p, 1);
	}
	return p;
}

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


// backup
// static float* conv2d_memsafe(float* x, int xw, int xh,
// 	float* w, float* b, int* desc,
// 	float xq_step, float wq_step,
// 	int& yw, int& yh, int& yn) {


// 	int c = desc[0];
// 	int n = desc[1];
// 	int k = desc[2];
// 	int s = desc[3];
// 	int p = desc[4];

// 	if (xq_step > 0 && ECC_ITER > 0) {
// 		quantize_gpu(xw * xh * c, xq_step, 11, true, x);
// 	}

// 	int xpw = xw + 2 * p;
// 	int xph = xh + 2 * p;
	
// 	float* x_padded = cuda_make_array(0, xpw * xph * c);
// 	padding_gpu(x, xw, xh, c, p, x_padded);
// 	cuda_free(x);

// 	float* x_mat = cuda_make_array(0, xpw * xph * c * k * k);
// 	unrolling_gpu(x_padded, xpw, xph, c, k, s, x_mat);
// 	cuda_free(x_padded); // x_padded expires

// 	yw = (xpw - k) / s + 1;
// 	yh = (xph - k) / s + 1;
// 	yn = n;

// 	int y_size = yw * yh;
// 	int f_size = k * k * c;
// 	float* y = cuda_make_array(0, y_size * n);
// 	tile_repeat_gpu(n, 1, y_size, b, 1, y, 1);

// 	int g = 4;

// 	float* xw_mat = cuda_make_array(0, y_size * f_size * g);
// 	for (int ni = 0; ni < n; ni += g) {
// 		if (xq_step > 0 && ECC_ITER > 0) {
// 			// distribute_mul_gpu(x_mat, w + ni * f_size, yw, yh, c, k, 1, xw_mat);
// 			APPROX_FUNC(x_mat, w + ni * f_size, yw, yh, c, k, g, xw_mat, ECC_ITER);
// 			scale_gpu(y_size * f_size * g, xq_step * wq_step, xw_mat, 1);
// 		} else {
// 			distribute_mul_gpu(x_mat, w + ni * f_size, yw, yh, c, k, g, xw_mat);
// 		}
// 		accumulate_gpu(y_size * g, f_size, xw_mat, 1, y + ni * y_size, 1);
// 	}

// 	cuda_free(x_mat); // x_mat expires
// 	cuda_free(xw_mat); // xw_mat expires

// 	return y;
// }

// another backup with request_workspace
// static void conv2d(float* x, int xw, int xh,
// 	float* w, float* b, int* desc,
// 	float xq_step, float wq_step,
// 	float* y, int& yw, int& yh, int& yn,
// 	int g) {

// 	int c = desc[0];
// 	int n = desc[1];
// 	int k = desc[2];
// 	int s = desc[3];
// 	int p = desc[4];

// 	g = g == 0 ? n : g;

// 	float* x_qtz = request_workspace(xw * xh * c, true);

// 	if (xq_step > 0 && ECC_ITER > 0) {
// 		quantize_gpu(xw * xh * c, xq_step, 11, true, x, 1, x_qtz, 1);
// 	} else {
// 		copy_gpu(xw * xh * c, x, 1, x_qtz, 1);
// 	}

// 	int xpw = xw + 2 * p;
// 	int xph = xh + 2 * p;
	
// 	float* x_padded = request_workspace(xpw * xph * c, true);
// 	padding_gpu(x_qtz, xw, xh, c, p, x_padded);

// 	float* x_mat = request_workspace(xpw * xph * c * k * k, true);
// 	unrolling_gpu(x_padded, xpw, xph, c, k, s, x_mat);

// 	yw = (xpw - k) / s + 1;
// 	yh = (xph - k) / s + 1;
// 	yn = n;

// 	int y_size = yw * yh;
// 	int f_size = k * k * c;
// 	fill_gpu(y_size * n, 0, y, 1);
// 	tile_repeat_gpu(n, 1, y_size, b, 1, y, 1);

// 	float* xw_mat = request_workspace(y_size * f_size * g, true);
// 	for (int ni = 0; ni < n; ni += g) {
// 		if (xq_step > 0 && ECC_ITER > 0) {
// 			// distribute_mul_gpu(x_mat, w + ni * f_size, yw, yh, c, k, 1, xw_mat);
// 			APPROX_FUNC(x_mat, w + ni * f_size, yw, yh, c, k, g, xw_mat, ECC_ITER);
// 			scale_gpu(y_size * f_size * g, xq_step * wq_step, xw_mat, 1);
// 		} else {
// 			distribute_mul_gpu(x_mat, w + ni * f_size, yw, yh, c, k, g, xw_mat);
// 		}
// 		accumulate_gpu(y_size * g, f_size, xw_mat, 1, y + ni * y_size, 1);
// 	}
// }

static void conv2d(float* x, int xw, int xh,
	float* w, float* b, int* desc,
	float xq_step, float wq_step,
	float* y, int& yw, int& yh, int& yn) {

	int c = desc[0];
	int n = desc[1];
	int k = desc[2];
	int s = desc[3];
	int p = desc[4];

	yn = n; yh = xh; yw = xw;

	float* x_qtz = request_workspace(xw * xh * c, true);

	if (xq_step > 0 && ECC_ITER > 0) {
		quantize_gpu(xw * xh * c, xq_step, 8, true, x, 1, x_qtz, 1);
		// print_array(x, 1, xh, xw);
		// print_array(x_qtz, 1, xh, xw);
	} else {
		copy_gpu(xw * xh * c, x, 1, x_qtz, 1);
	}

	int y_size = xw * xh;
	int f_size = k * k * c;
	fill_gpu(y_size * n, 0, y, 1);
	tile_repeat_gpu(n, 1, y_size, b, 1, y, 1);

	float* xw_mat = request_workspace(y_size * c, true);
	for (int ni = 0; ni < n; ni += 1) {
		if (xq_step > 0 && ECC_ITER > 0) {
			// distribute_mul_gpu(x_mat, w + ni * f_size, yw, yh, c, k, 1, xw_mat);
			APPROX_FUNC(x_qtz, w + ni * f_size, xw, xh, c, k, xw_mat, ECC_ITER);
			scale_gpu(y_size * c, xq_step * wq_step, xw_mat, 1);
		} else {
			conv2d_chunk_mul_gpu(x_qtz, w + ni * f_size, xw, xh, c, k, xw_mat);
		}
		accumulate_gpu(y_size, c, xw_mat, 1, y + ni * y_size, 1);
	}
}

// make it inline modifier or return? currently inline
static void mean_shift(float* x, int xw, int xh, float sign) {
	float* _mean = new float[3]{sign * 0.4488f * 255.0f, sign * 0.4371f * 255.0f, sign * 0.4040f * 255.0f};
	float* mean = request_workspace(3, true);
	cuda_push_array(mean, _mean, 3);

	broadcast_add_gpu(3, xw * xh, x, mean, x);

	delete[] _mean;
}

static void rgb2y(float* x, int xw, int xh, float* y) {
	float* _weight = new float[3]{65.738f / 255.0f / 256.0f, 129.057f / 255.0f / 256.0f, 25.064f / 255.0f / 256.0f};
	float* weight = request_workspace(3, true);
	cuda_push_array(weight, _weight, 3);

	float* _bias = new float[1]{16.0f / 256.0f};
	float* bias = request_workspace(1, true);
	cuda_push_array(bias, _bias, 3);

	int _config[5] = {3, 1, 1, 1, 0};

	int yw, yh, yn;

	conv2d(x, xw, xh, weight, bias, _config, 0, 0, y, yw, yh, yn);

	delete[] _weight;
	delete[] _bias;
}

static float forward(float* im, int imw, int imh, float* gt, int gtw, int gth,
	int* _head, int* _resb, int* _body, int* _upsl, int* _tail, int _resr,
	std::unordered_map<std::string, float*> weights, std::unordered_map<std::string, float*> biases,
	std::unordered_map<std::string, float> wq_steps, std::unordered_map<std::string, float> xq_steps,
	float*& z_im_c) {

	std::string layer_name;
	float *x, *y, *z, *x_head;
	int xn, xh, xw;
	int yn, yh, yw;
	int zn, zh, zw;

	x = node["x"];
	y = node["y"];
	z = node["z"];
	x_head = node["x_head"];

	xn = 3; xh = imh; xw = imw;
	copy_gpu(xn * xh * xw, im, 1, x, 1);

	// mean_shift
	mean_shift(x, xw, xh, -1); // output saved to im itself, no return

	// head
	layer_name = "head.0";
	conv2d(x, xw, xh, weights[layer_name], biases[layer_name], _head, xq_steps[layer_name], wq_steps[layer_name], z, zw, zh, zn);

	copy_gpu(zw*zh*zn, z, 1, x_head, 1); // saving for post-residual use
	copy_gpu(zw*zh*zn, z, 1, x, 1);
	xn = zn; xh = zh; xw = zw;
	
	// res repeat
	for (int i = 0; i < _resr; ++i) {
		layer_name = "body." + std::to_string(i) + ".body.0";
		// std::cout << "[INFO] layer: " << layer_name << std::endl;

		conv2d(x, xw, xh, weights[layer_name], biases[layer_name], _resb, xq_steps[layer_name], wq_steps[layer_name], y, yw, yh, yn);
		min_gpu(yw * yh * yn, 0, y, 1);

		layer_name = "body." + std::to_string(i) + ".body.2";
		// std::cout << "[INFO] layer: " << layer_name << std::endl;

		conv2d(y, yw, yh, weights[layer_name], biases[layer_name], _resb, xq_steps[layer_name], wq_steps[layer_name], z, zw, zh, zn);
		axpy_gpu(zn * zh * zw, 0.1, z, 1, x, 1);
		xn = zn; xh = zh; xw = zw;
	}

	// res body last
	layer_name = "body." + std::to_string(_resr);
	// std::cout << "[INFO] layer: " << layer_name << std::endl;
	conv2d(x, xw, xh, weights[layer_name], biases[layer_name], _body, xq_steps[layer_name], wq_steps[layer_name], y, yw, yh, yn);
	axpy_gpu(yn * yh * yw, 1, x_head, 1, y, 1);

	// upsl
	layer_name = "tail.0.0";
	// std::cout << "[INFO] layer: " << layer_name << std::endl;
	conv2d(y, yw, yh, weights[layer_name], biases[layer_name], _upsl, xq_steps[layer_name], wq_steps[layer_name], z, zw, zh, zn);
	
	
	pixel_shuffle_gpu(z, zw, zh, zn / 4, 2, x);
	xn = zn / 4; xh = zh * 2; xw = zw * 2;

	// // tail conv 1
	layer_name = "tail.1";
	// std::cout << "[INFO] layer: " << layer_name << std::endl;
	conv2d(x, xw, xh, weights[layer_name], biases[layer_name], _tail, xq_steps[layer_name], wq_steps[layer_name], y, yw, yh, yn);
	
	mean_shift(y, yw, yh, 1); // output saved to im itself, no return
	
	rgb2y(y, xw, xh, z);
	rgb2y(gt, gtw, gth, y);

	axpy_gpu(gtw*gth, -1, y, 1, z, 1);
	pow_gpu(gtw*gth, 2, z, 1);
	float sum = 0;
	float* t = cuda_make_array(0, 1);
	float _t;
	for (int hi = 2; hi < gth - 2; ++hi) {
		fill_gpu(1, 0, t, 1);
		accumulate_gpu(1, gtw-4, z+hi*gtw+2, 1, t, 1);
		cuda_pull_array(t, &_t, 1);
		sum += _t;
	}
	float mean = sum / ((gth-4)*(gtw-4));

	cuda_free(t);
	return -10 * log10(mean);
}

static int parse(int argc, char** argv) {
	MODEL_PATH = argv[2];
	
	if (0 == strcmp(argv[3], "set14")) {
		N_IMAGE = 14;
		DATA_DIR = "data/set14/";
	} else if (0 == strcmp(argv[3], "set5")) {
		N_IMAGE = 5;
		DATA_DIR = "data/set5/";
	} else if (0 == strcmp(argv[3], "b100")) {
		N_IMAGE = 100;
		DATA_DIR = "data/b100/";
	} else if (0 == strcmp(argv[3], "urban100")) {
		N_IMAGE = 100;
		DATA_DIR = "data/urban100/";
	} else if (0 == strcmp(argv[3], "dummy")) {
		N_IMAGE = 1;
		DATA_DIR = "data/dummy_20x20/";
	} else if (0 == strcmp(argv[3], "dummy_large")) {
		N_IMAGE = 1;
		DATA_DIR = "data/dummy_large/";
	} else {
		std::cout << "[ERRO] unrecognized dataset: " << argv[3] << std::endl;
		return 1;
	}

	if (argc >= 5) {
		ECC_ITER = std::stoi(argv[4]);
		std::cout << "[INFO] setting ECC_ITER = " << ECC_ITER << std::endl;
	}

	if (argc >= 6) {
		if (0 == strcmp(argv[5], "--ilm")) {
			std::cout << "[INFO] using ILM approximation" << std::endl;
			APPROX_FUNC = conv2d_chunk_approximate_ilm_iterative_gpu;
		} else if (0 == strcmp(argv[5], "--ori")) {
			std::cout << "[INFO] using original approximation" << std::endl;
			APPROX_FUNC = conv2d_chunk_approximate_one_side_iterative_gpu;
		} else if (0 == strcmp(argv[5], "--signedap")) {
			std::cout << "[INFO] using signedap approximation" << std::endl;
			APPROX_FUNC = conv2d_chunk_approximate_signedap_iterative_gpu;
		} else if (0 == strcmp(argv[5], "--one")) {
			std::cout << "[INFO] using custom (min-error) approximation" << std::endl;
			APPROX_FUNC = conv2d_chunk_approximate_iterative_gpu;
		} else if (0 == strcmp(argv[5], "--lsbskip")) {
			std::cout << "[INFO] using lsbskip approximation" << std::endl;
			APPROX_FUNC = conv2d_chunk_approximate_lsbskip_iterative_gpu;
		}
	}

	if (argc >= 7) {
		WP = std::stof(argv[6]);
		std::cout << "[INFO] setting WP = " << WP << std::endl;
	}
	
	return 0;
}

int malm_edsr_2x(int argc, char** argv) {
	if (parse(argc, argv)) {
		std::cout << "[ERRO] parsing error" << std::endl;
		return 1;
	}
	
	int _head[5] = {  3, 256, 3, 1, 1};
	int _resb[5] = {256, 256, 3, 1, 1};
	int _body[5] = {256, 256, 3, 1, 1};
	int _upsl[5] = {256,1024, 3, 1, 1};
	int _tail[5] = {256,   3, 3, 1, 1};
	int _resr = 32;

	workspace = cuda_make_array(0, ws_size); // about 4GB
	for (auto name: node_names) {
		node[name] = cuda_make_array(0, node_size);
	}

	std::vector<std::string> layer_names = {"head.0", "body.0.body.0", "body.0.body.2", "body.1.body.0", "body.1.body.2", "body.2.body.0", "body.2.body.2", "body.3.body.0", "body.3.body.2", "body.4.body.0", "body.4.body.2", "body.5.body.0", "body.5.body.2", "body.6.body.0", "body.6.body.2", "body.7.body.0", "body.7.body.2", "body.8.body.0", "body.8.body.2", "body.9.body.0", "body.9.body.2", "body.10.body.0", "body.10.body.2", "body.11.body.0", "body.11.body.2", "body.12.body.0", "body.12.body.2", "body.13.body.0", "body.13.body.2", "body.14.body.0", "body.14.body.2", "body.15.body.0", "body.15.body.2", "body.16.body.0", "body.16.body.2", "body.17.body.0", "body.17.body.2", "body.18.body.0", "body.18.body.2", "body.19.body.0", "body.19.body.2", "body.20.body.0", "body.20.body.2", "body.21.body.0", "body.21.body.2", "body.22.body.0", "body.22.body.2", "body.23.body.0", "body.23.body.2", "body.24.body.0", "body.24.body.2", "body.25.body.0", "body.25.body.2", "body.26.body.0", "body.26.body.2", "body.27.body.0", "body.27.body.2", "body.28.body.0", "body.28.body.2", "body.29.body.0", "body.29.body.2", "body.30.body.0", "body.30.body.2", "body.31.body.0", "body.31.body.2", "body.32", "tail.0.0", "tail.1"};

	std::unordered_map<std::string, float> wq_steps;
	std::unordered_map<std::string, float> xq_steps;
	
	for (auto name: layer_names) {
		wq_steps[name] = 0; xq_steps[name] = 0;
	}

	wq_steps["head.0"] = 1.0 / (1 << 12); xq_steps["head.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.0.body.0"] = 1.0 / (1 << 12); xq_steps["body.0.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.0.body.2"] = 1.0 / (1 << 12); xq_steps["body.0.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.1.body.0"] = 1.0 / (1 << 12); xq_steps["body.1.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.1.body.2"] = 1.0 / (1 << 11); xq_steps["body.1.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.2.body.0"] = 1.0 / (1 << 12); xq_steps["body.2.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.2.body.2"] = 1.0 / (1 << 11); xq_steps["body.2.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.3.body.0"] = 1.0 / (1 << 12); xq_steps["body.3.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.3.body.2"] = 1.0 / (1 << 12); xq_steps["body.3.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.4.body.0"] = 1.0 / (1 << 12); xq_steps["body.4.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.4.body.2"] = 1.0 / (1 << 12); xq_steps["body.4.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.5.body.0"] = 1.0 / (1 << 11); xq_steps["body.5.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.5.body.2"] = 1.0 / (1 << 11); xq_steps["body.5.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.6.body.0"] = 1.0 / (1 << 11); xq_steps["body.6.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.6.body.2"] = 1.0 / (1 << 12); xq_steps["body.6.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.7.body.0"] = 1.0 / (1 << 12); xq_steps["body.7.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.7.body.2"] = 1.0 / (1 << 12); xq_steps["body.7.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.8.body.0"] = 1.0 / (1 << 12); xq_steps["body.8.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.8.body.2"] = 1.0 / (1 << 12); xq_steps["body.8.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.9.body.0"] = 1.0 / (1 << 12); xq_steps["body.9.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.9.body.2"] = 1.0 / (1 << 12); xq_steps["body.9.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.10.body.0"] = 1.0 / (1 << 11); xq_steps["body.10.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.10.body.2"] = 1.0 / (1 << 11); xq_steps["body.10.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.11.body.0"] = 1.0 / (1 << 12); xq_steps["body.11.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.11.body.2"] = 1.0 / (1 << 12); xq_steps["body.11.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.12.body.0"] = 1.0 / (1 << 12); xq_steps["body.12.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.12.body.2"] = 1.0 / (1 << 11); xq_steps["body.12.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.13.body.0"] = 1.0 / (1 << 12); xq_steps["body.13.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.13.body.2"] = 1.0 / (1 << 11); xq_steps["body.13.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.14.body.0"] = 1.0 / (1 << 12); xq_steps["body.14.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.14.body.2"] = 1.0 / (1 << 11); xq_steps["body.14.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.15.body.0"] = 1.0 / (1 << 12); xq_steps["body.15.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.15.body.2"] = 1.0 / (1 << 11); xq_steps["body.15.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.16.body.0"] = 1.0 / (1 << 12); xq_steps["body.16.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.16.body.2"] = 1.0 / (1 << 11); xq_steps["body.16.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.17.body.0"] = 1.0 / (1 << 12); xq_steps["body.17.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.17.body.2"] = 1.0 / (1 << 12); xq_steps["body.17.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.18.body.0"] = 1.0 / (1 << 12); xq_steps["body.18.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.18.body.2"] = 1.0 / (1 << 12); xq_steps["body.18.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.19.body.0"] = 1.0 / (1 << 12); xq_steps["body.19.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.19.body.2"] = 1.0 / (1 << 12); xq_steps["body.19.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.20.body.0"] = 1.0 / (1 << 12); xq_steps["body.20.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.20.body.2"] = 1.0 / (1 << 12); xq_steps["body.20.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.21.body.0"] = 1.0 / (1 << 12); xq_steps["body.21.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.21.body.2"] = 1.0 / (1 << 12); xq_steps["body.21.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.22.body.0"] = 1.0 / (1 << 12); xq_steps["body.22.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.22.body.2"] = 1.0 / (1 << 12); xq_steps["body.22.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.23.body.0"] = 1.0 / (1 << 12); xq_steps["body.23.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.23.body.2"] = 1.0 / (1 << 11); xq_steps["body.23.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.24.body.0"] = 1.0 / (1 << 11); xq_steps["body.24.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.24.body.2"] = 1.0 / (1 << 11); xq_steps["body.24.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.25.body.0"] = 1.0 / (1 << 11); xq_steps["body.25.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.25.body.2"] = 1.0 / (1 << 11); xq_steps["body.25.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.26.body.0"] = 1.0 / (1 << 12); xq_steps["body.26.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.26.body.2"] = 1.0 / (1 << 12); xq_steps["body.26.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.27.body.0"] = 1.0 / (1 << 11); xq_steps["body.27.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.27.body.2"] = 1.0 / (1 << 11); xq_steps["body.27.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.28.body.0"] = 1.0 / (1 << 11); xq_steps["body.28.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.28.body.2"] = 1.0 / (1 << 11); xq_steps["body.28.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.29.body.0"] = 1.0 / (1 << 11); xq_steps["body.29.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.29.body.2"] = 1.0 / (1 << 11); xq_steps["body.29.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.30.body.0"] = 1.0 / (1 << 11); xq_steps["body.30.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.30.body.2"] = 1.0 / (1 << 11); xq_steps["body.30.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.31.body.0"] = 1.0 / (1 << 11); xq_steps["body.31.body.0"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.31.body.2"] = 1.0 / (1 << 10); xq_steps["body.31.body.2"] = 2.0;//1.0 / (1 << 0);
	wq_steps["body.32"] = 1.0 / (1 << 12); xq_steps["body.32"] = 2.0;//1.0 / (1 << 0);
	wq_steps["tail.0.0"] = 1.0 / (1 << 10); xq_steps["tail.0.0"] = 4.0;//1.0 / (1 << 0);
	wq_steps["tail.1"] = 1.0 / (1 << 12); xq_steps["tail.1"] = 4.0;//1.0 / (1 << 0);
	
	std::unordered_map<std::string, float*> weights;
	std::unordered_map<std::string, float*> biases;

	for (auto name: layer_names) {
		int c = 0, n = 0, k = 0;

		if (name.substr(0, 4) == "head") {
			c = _head[0];
			n = _head[1];
			k = _head[2];
		} else if (name.substr(0, 4) == "body") {
			c = _resb[0];
			n = _resb[1];
			k = _resb[2];
		} else if (name.substr(0, 4) == "tail") {
			if (name[5] == '0') {
				c = _upsl[0];
				n = _upsl[1];
				k = _upsl[2];
			} else if (name[5] == '1') {
				c = _tail[0];
				n = _tail[1];
				k = _tail[2];
			}
		}

		std::string data_file_name;
		FILE* f;

		data_file_name = MODEL_PATH + "/" + name + ".bias";
		f = fopen(data_file_name.c_str(), "r");
		if (!f) {
			std::cout << "[ERRO] cannot load model from " + data_file_name << std::endl;
			return 1;
		}

		int bias_size = n;
		float* _bias = new float[bias_size];
		fread(_bias, sizeof(float), bias_size, f);
		biases[name] = cuda_make_array(_bias, bias_size);
		delete[] _bias;
		fclose(f);

		data_file_name = MODEL_PATH + "/" + name + ".weight";
		f = fopen(data_file_name.c_str(), "r");
		if (!f) {
			std::cout << "[ERRO] cannot load model from " + data_file_name << std::endl;
			return 1;
		}

		int weight_size = c * n * k * k;
		float* _weight = new float[weight_size];
		fread(_weight, sizeof(float), weight_size, f);
		weights[name] = cuda_make_array(_weight, weight_size);

		// add quantization
		if (wq_steps[name] > 0 && ECC_ITER > 0) {
			quantize_compensate_wp_gpu(weight_size, wq_steps[name], 11, WP, true, weights[name]);
			// print_array(weights[name], 1, 1, k*k);
		}
		delete[] _weight;
		fclose(f);
	}


	// // load images
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

		std::cout << "[DEBG] " << "im: " << imw << " x " << imh << std::endl;

		int im_size = 3 * imw * imh;
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

		// std::cout << "[DEBG] " << "gt: " << gtw << " x " << gth << std::endl;

		int gt_size = 3 * gtw * gth;
		float* _gt = new float[gt_size];
		fread(_gt, sizeof(float), gt_size, f);
		float* gt = cuda_make_array(_gt, gt_size);
		delete[] _gt;

		fclose(f);

		float* output_c;

		// forwarding
		float psnr = forward(im, imw, imh, gt, gtw, gth, _head, _resb, _body, _upsl, _tail, _resr, weights, biases, wq_steps, xq_steps, output_c);
		// forwarding end

		// f = fopen((DATA_DIR + "_" + std::to_string(ECC_ITER) + "_" + std::to_string(WP) + "/ou_" + std::to_string(i)).c_str(), "wb");
		// fwrite(output_c, sizeof(float), gtw * gth + 2, f);
		// fclose(f);
		// delete[] output_c;

		psnr_mean += psnr;
		std::cout << psnr << std::endl;
		
		cuda_free(im);
		cuda_free(gt);
	}
	std::cout << psnr_mean / N_IMAGE << std::endl;

	// cleanup:
	for (auto name: layer_names) {
		cuda_free(biases[name]);
		cuda_free(weights[name]);
	}
	for (auto name: node_names) {
		cuda_free(node[name]);
	}

	cuda_free(workspace);

	return 0;
}