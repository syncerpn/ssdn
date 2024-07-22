#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "cuda.h"
#include "utils.h"

__device__ float three_way_max_kernel(float a, float b, float c) {
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c);
}

__device__ float three_way_min_kernel(float a, float b, float c) {
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

__global__ void rgb_to_hsv_kernel(float* im_data, int im_w, int im_h) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= im_w*im_h) return;
	float r = im_data[0*im_h*im_w + index];
	float g = im_data[1*im_h*im_w + index];
	float b = im_data[2*im_h*im_w + index];
	float max = three_way_max_kernel(r,g,b);
	float min = three_way_min_kernel(r,g,b);
	float delta = max - min;
	float v = max;
	float s, h;
	if (max == 0) {
		s = 0;
		h = 0;
	} else {
		s = delta/max;
		if (r == max) {
			h = (g - b) / delta;
		} else if (g == max) {
			h = 2 + (b - r) / delta;
		} else {
			h = 4 + (r - g) / delta;
		}
		if (h < 0) h += 6;
		h = h/6.;
	}
	im_data[0*im_h*im_w + index] = h;
	im_data[1*im_h*im_w + index] = s;
	im_data[2*im_h*im_w + index] = v;
}

void rgb_to_hsv_gpu(float* im_data, int w, int h, int c) {
	if (c != 3) {
		fprintf(stderr, "rgb_to_hsv_kernel: failed, num channel should be 3\n");
	}
	rgb_to_hsv_kernel<<<cuda_gridsize(h*w), BLOCK>>>(im_data, w, h);
	check_error(cudaPeekAtLastError());
}

__global__ void hsv_to_rgb_kernel(float* im_data, int im_w, int im_h) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= im_w*im_h) return;
	float h = 6 * im_data[0*im_h*im_w + index];
	float s = im_data[1*im_h*im_w + index];
	float v = im_data[2*im_h*im_w + index];
	float r,g,b;
	if (s == 0) {
		r = g = b = v;
	} else {
		int floor_h = floor(h);
		float f = h - floor_h;
		float p = v * (1-s);
		float q = v * (1-s*f);
		float t = v * (1-s*(1-f));
		if (floor_h == 0) {
			r = v; g = t; b = p;
		} else if (floor_h == 1) {
			r = q; g = v; b = p;			
		} else if (floor_h == 2) {
			r = p; g = v; b = t;
		} else if (floor_h == 3) {
			r = p; g = q; b = v;
		} else if (floor_h == 4) {
			r = t; g = p; b = v;
		} else {
			r = v; g = p; b = q;
		}
	}
	im_data[0*im_h*im_w + index] = r;
	im_data[1*im_h*im_w + index] = g;
	im_data[2*im_h*im_w + index] = b;
}

void hsv_to_rgb_gpu(float* im_data, int w, int h, int c) {
	if (c != 3) {
		fprintf(stderr, "hsv_to_rgb_kernel: failed, num channel should be 3\n");
	}
	hsv_to_rgb_kernel<<<cuda_gridsize(h*w), BLOCK>>>(im_data, w, h);
	check_error(cudaPeekAtLastError());
}

__global__ void solarize_image_kernel(float* im_data, int w, int h, int c, float threshold) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= c*h*w) return;
	if (im_data[index] > threshold) im_data[index] = 1 - im_data[index];
}

void solarize_image_gpu(float* im_data, int w, int h, int c, float threshold) {
	solarize_image_kernel<<<cuda_gridsize(c*h*w), BLOCK>>>(im_data, w, h, c, threshold);
	check_error(cudaPeekAtLastError());
}

__global__ void posterize_image_kernel(float* im_data, int w, int h, int c, int levels) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= c*h*w) return;
	float step = 1./(levels - 1);
	im_data[index] = (round((im_data[index] + step / 2.f) / step) - 0.5) * step;
	im_data[index] = im_data[index] > 1 ? 1 : (im_data[index] < 0 ? 0 : im_data[index]);
}

void posterize_image_gpu(float* im_data, int w, int h, int c, int levels) {
	posterize_image_kernel<<<cuda_gridsize(c*h*w), BLOCK>>>(im_data, w, h, c, levels);
	check_error(cudaPeekAtLastError());
}

__global__ void resize_image_kernel(float* input, int iw, int ih, float* output, int ow, int oh, int oc) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= ow * oh * oc) return;
	output[index] = 0;
	int ox = index % ow;
	index = (index - ox) / ow;
	int oy = index % oh;
	int oz = (index - oy) / oh;
	index = index * ow + ox;

	float w_scale = (float)iw / ow;
	float h_scale = (float)ih / oh;
	
	float ix_s = ox * w_scale;
	int ix_s_floor = floor(ix_s);
	float ix_e = (ox + 1) * w_scale;
	int ix_e_ceil = ceil(ix_e);
	
	float iy_s = oy * h_scale;
	int iy_s_floor = floor(iy_s);
	float iy_e = (oy + 1) * h_scale;
	int iy_e_ceil = ceil(iy_e);

	int i, j;
	for (j = iy_s_floor; j < iy_e_ceil; ++j) {
		for (i = ix_s_floor; i < ix_e_ceil; ++i) {
			int in_index = oz*ih*iw + j*iw + i;
			float delta_y = 1;
			float delta_x = 1;
			if (j == iy_s_floor) delta_y = (float)iy_s_floor + 1 - iy_s;
			if (j == iy_e_ceil - 1) delta_y = iy_e - (float)iy_e_ceil + 1;
			if (i == ix_s_floor) delta_x = (float)ix_s_floor + 1 - ix_s;
			if (i == ix_e_ceil - 1) delta_x = ix_e - (float)ix_e_ceil + 1;
			delta_x = delta_x < w_scale ? delta_x : w_scale;
			delta_y = delta_y < h_scale ? delta_y : h_scale;
			output[index] += delta_x * delta_y * input[in_index];
		}
	}
	output[index] /= (w_scale * h_scale);
}

void resize_image_gpu(float* input, int iw, int ih, float* output, int ow, int oh, int oc) {
	resize_image_kernel<<<cuda_gridsize(oc*oh*ow), BLOCK>>>(input, iw, ih, output, ow, oh, oc);
	check_error(cudaPeekAtLastError());
}