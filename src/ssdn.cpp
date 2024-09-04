#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "ssdn.h"

void conv2d_bn(float* x, int xw, int xh,
	float* w, float* scale, float* b, int* desc,
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

int run_sim_fast_approx_ma(std::string model_path, float wp) {
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
		std::cout << "[INFO] load layer " << i << std::endl;
		std::string data_file_name = model_path + "/layer_" + std::to_string(i);
		FILE* f = fopen(data_file_name.c_str(), "r");
		if (!f) {
			std::cout << "[ERRO] cannot load model from " + model_path << std::endl;
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
		if (wq_steps[i] > 0) {
			// quantize_gpu(weight_size, wq_steps[i], 11, true, weights[i]);
			// compensate_wp_gpu(weight_size, wp, weights[i]);
			// max_gpu(weight_size, (1 << 10) - 1, weights[i]);
			// min_gpu(weight_size, -1 << 10, weights[i]);

			quantize_compensate_wp_gpu(weight_size, wq_steps[i], 9, wp, true, weights[i]);

			// quantize_gpu(weight_size, wq_steps[i], 11, true, weights[i]);
			// compensate_log_gpu(weight_size, wp, 1 << 10, weights[i]);
			// max_gpu(weight_size, (1 << 10) - 1, weights[i]);
			// min_gpu(weight_size, -1 << 10, weights[i]);
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
	return 0;
}

void load_cifar10(const char *filename, uint8_t *labels, uint8_t *images) {
    FILE *file = fopen(filename, "rb");  // Open file in binary mode
    if (!file) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
    }

    for (int i = 0; i < 10000; ++i) {
        // Read label (1 byte)
        if (fread(&labels[i], 1, 1, file) != 1) {
            fprintf(stderr, "Error reading label from file: %s\n", filename);
            fclose(file);
        }

        // Read image data (3072 bytes)
        if (fread(&images[i * 32*32*3], 32*32*3, 1, file) != 1) {
            fprintf(stderr, "Error reading image data from file: %s\n", filename);
            fclose(file);
        }
    }

    fclose(file);
}

void print_array(float* x, int n, int m, int k) {
	for (int ni = 0; ni < n; ++ni) {
		for (int mi = 0; mi < m; ++mi) {
			for (int ki = 0; ki < k; ++ki) {
				std::cout << x[ni*m*k + mi*k + ki] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

void print_array_u(uint8_t* x, int n, int m, int k) {
	for (int ni = 0; ni < n; ++ni) {
		for (int mi = 0; mi < m; ++mi) {
			for (int ki = 0; ki < k; ++ki) {
				std::cout << x[ni*m*k + mi*k + ki] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

float forward_cls_single(float* im, float* gt,
	int** layers, int n_layer, float** weights, float** scales, float** biases,
	float* wq_steps, float* xq_steps, float** workspace) {

	int zw, zh, zn;
	float* z = workspace[3];

	copy_gpu(32*32*3, im, 1, z, 1);

	float* x = z;
	int xw = 32;
	int xh = 32;

	for (int li = 0; li < n_layer; ++li) {
		std::cout << "[INFO] layer " << li;
		conv2d_bn(x, xw, xh, weights[li], scales[li], biases[li], layers[li], xq_steps[li], wq_steps[li], zw, zh, zn, workspace);
		if (li != n_layer-1) {
			min_gpu(zw*zh*zn, 0, z, 1);
		}
		x = z;
		xw = zw;
		xh = zh;
		std::cout << " done" << std::endl;
	}
}

int run_sim_fast_approx_ma_cls(std::string model_path, float wp) {
	int _layers[95] = { 3, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 16, 3, 1, 1, 16, 32, 3, 2, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 32, 3, 1, 1, 32, 64, 3, 2, 1, 64, 64, 3, 1, 1, 64, 64, 3, 1, 1, 64, 64, 3, 1, 1, 64, 64, 3, 1, 1, 64, 64, 3, 1, 1};

	int **layers = new int*[19];
	for (int i = 0; i < 19; ++i) {
		layers[i] = new int[5];
	}
	for (int i = 0; i < 19; ++i) {
		for (int j = 0; j < 5; ++j) {
			layers[i][j] = _layers[i*5+j];
		}
	}

	float wq_steps[19] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	float xq_steps[19] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	const size_t SPA_SIZE_MAX = 16*16*32*32;
	const size_t K_MAX = 3;

	float** weights = new float*[19];
	float** scales = new float*[19];
	float** biases = new float*[19];
	float** workspace = new float*[4];

	std::cout << "[INFO] allocating GPU mem for processing" << std::endl;
	workspace[0] = cuda_make_array(0, SPA_SIZE_MAX * K_MAX * K_MAX); // x_padded
	workspace[1] = cuda_make_array(0, SPA_SIZE_MAX * K_MAX * K_MAX); // x_mat
	workspace[2] = cuda_make_array(0, SPA_SIZE_MAX * K_MAX * K_MAX); // xw_mat
	workspace[3] = cuda_make_array(0, SPA_SIZE_MAX * K_MAX * K_MAX); // y
	std::cout << "[INFO] allocation finished" << std::endl;
	// load model
	for (int i = 0; i < 19; ++i) {
		std::cout << "[INFO] load layer " << 2*i << std::endl;
		std::string data_file_name = model_path + "/layer_" + std::to_string(2*i);
		FILE* f = fopen(data_file_name.c_str(), "r");
		if (!f) {
			std::cout << "[ERRO] cannot load model from " + model_path << std::endl;
			return 1;
		}
		
		std::cout << "[INFO] load batchnorm layer " << 2*i+1 << std::endl;
		data_file_name = model_path + "/layer_" + std::to_string(2*i+1);
		FILE* f2 = fopen(data_file_name.c_str(), "r");
		if (!f2) {
			std::cout << "[ERRO] cannot load model from " + model_path << std::endl;
			return 1;
		}

		int c = layers[i][0];
		int n = layers[i][1];
		int k = layers[i][2];

		int weight_size = c * n * k * k;
		float* _weight = new float[weight_size];
		fread(_weight, sizeof(float), weight_size, f);
		weights[i] = cuda_make_array(_weight, weight_size);
		// print_array(_weight, n, c, k * k);

		// quantize_compensate_wp_gpu(weight_size, wq_step, 11, wp, true, weights[i]);

		delete[] _weight;

		int scale_bias_size = n;

		float* _bias = new float[scale_bias_size];
		fread(_bias, sizeof(float), scale_bias_size, f2);
		biases[i] = cuda_make_array(_bias, scale_bias_size);
		delete[] _bias;

		float* _scale = new float[scale_bias_size];
		fread(_scale, sizeof(float), scale_bias_size, f2);
		scales[i] = cuda_make_array(_scale, scale_bias_size);
		delete[] _scale;

		fclose(f);
		fclose(f2);
	}

	float* im = cuda_make_array(0, 10000 * 32 * 32 * 3);
	float* gt = cuda_make_array(0, 10000 * 1);
	
    uint8_t* im_u = (uint8_t*)malloc(10000 * 32 * 32 * 3);
	uint8_t* gt_u = (uint8_t*)malloc(10000 * 1);

	float* im_f = (float*)malloc(10000 * 32 * 32 * 3 * sizeof(float));
	float* gt_f = (float*)malloc(10000 * 1 * sizeof(float));

    load_cifar10("./data/cifar10/test_batch.bin", gt_u, im_u);
    for (int i = 0; i < 10000 * 32 * 32 * 3; ++i) {
    	im_f[i] = (float)(im_u[i]) / 256.0;
    }
	for (int i = 0; i < 10000 * 1; ++i) {
		gt_f[i] = (float)(gt_u[i]) / 1.0;
	}

	cuda_push_array(im, im_f, 10000 * 32 * 32 * 3);
	cuda_push_array(gt, gt_f, 10000 * 1);

	for (int i = 0; i < 1; ++i) {
		float acc = forward_cls_single(im + i*32*32*3, gt + i, layers, 19, weights, scales, biases, wq_steps, xq_steps, workspace);
	}

	// // load images
	// float acc_mean = 0;
	// for (int i = 0; i < 10000; ++i) {
	// 	std::string data_file_name;
	// 	FILE* f;
	// 	float wf, hf;
		
	// 	// read input image
	// 	data_file_name = "./data/imd_" + std::to_string(i);
	// 	f = fopen(data_file_name.c_str(), "r");
		
	// 	fread(&wf, sizeof(float), 1, f);
	// 	fread(&hf, sizeof(float), 1, f);
	// 	int imw = int(wf);
	// 	int imh = int(hf);

	// 	int im_size = imw * imh;
	// 	float* _im = new float[im_size];
	// 	fread(_im, sizeof(float), im_size, f);
	// 	cuda_push_array(im, _im, im_size);
	// 	delete[] _im;

	// 	fclose(f);

	// 	// read ground truth image
	// 	data_file_name = "./data/gtd_" + std::to_string(i);
	// 	f = fopen(data_file_name.c_str(), "r");
		
	// 	fread(&wf, sizeof(float), 1, f);
	// 	fread(&hf, sizeof(float), 1, f);
	// 	int gtw = int(wf);
	// 	int gth = int(hf);

	// 	int gt_size = gtw * gth;
	// 	float* _gt = new float[gt_size];
	// 	fread(_gt, sizeof(float), gt_size, f);
	// 	cuda_push_array(gt, _gt, gt_size);
	// 	delete[] _gt;

	// 	fclose(f);

	// 	// forwarding
	// 	float acc = forward(im, imw, imh, gt, gtw, gth, layers, 8, weights, biases, wq_steps, xq_steps, workspace);
	// 	acc_mean += acc;
	// }
	// std::cout << acc_mean / 10000 << std::endl;

	cuda_free(im);
	cuda_free(gt);
	cuda_free(workspace[0]);
	cuda_free(workspace[1]);
	cuda_free(workspace[2]);
	cuda_free(workspace[3]);

	for (int i = 0; i < 19; ++i) {
		cuda_free(weights[i]);
		cuda_free(biases[i]);
		cuda_free(scales[i]);
	}
	for (int i = 0; i < 19; ++i) {
		delete[] layers[i];
	}
	delete[] im_u;
	delete[] gt_u;
	delete[] layers;
	delete[] workspace;
	delete[] weights;
	delete[] biases;
	delete[] scales;
	return 0;
}

int main(int argc, char** argv) {
	if (argc == 1) {
		std::cout << "[ERRO] model path is required" << std::endl;
		return 1;
	}
	if (0 == strcmp(argv[1], "sr")) {
		for (int i = 0; i < 1000; ++i) {
			float wp = i / 1000.0;
			std::cout << "[INFO] wp = " << wp << std::endl;
			if (run_sim_fast_approx_ma(argv[2], wp)) {
				std::cout << "[ERRO] simulation finished with error(s)" << std::endl;
				return 2;
			}
		}
		return 0;
	} else if (0 == strcmp(argv[1], "cls")) {
		float wp = 0.1;
		if (run_sim_fast_approx_ma_cls(argv[2], wp)) {
			std::cout << "[ERRO] simulation finished with error(s)" << std::endl;
			return 2;
		}
		return 0;
	}
}