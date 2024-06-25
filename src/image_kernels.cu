#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "image.h"
#include "cuda.h"
#include "utils.h"
}

__global__ void flip_image_horizontal_kernel(float* im_data, int w, int h, int c) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	int i = index % w;
	index = (index - i) / w;
	int j = index % h;
	int k = (index - j) / h;
	if (i >= w/2) return;
	float tmp = im_data[k*h*w + j*w + i];
	im_data[k*h*w + j*w + i] = im_data[k*h*w + j*w + w-1-i];
	im_data[k*h*w + j*w + w-1-i] = tmp;
	return;
}

__global__ void flip_image_vertical_kernel(float* im_data, int w, int h, int c) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	int i = index % w;
	index = (index - i) / w;
	int j = index % h;
	int k = (index - j) / h;
	if (j >= h/2) return;
	float tmp = im_data[k*h*w + j*w + i];
	im_data[k*h*w + j*w + i] = im_data[k*h*w + (h-1-j)*w + i];
	im_data[k*h*w + (h-1-j)*w + i] = tmp;
	return;
}

void flip_image_x_gpu(float* im_data, int w, int h, int c, int hflip, int vflip) {
	if (hflip) {
		flip_image_horizontal_kernel<<<cuda_gridsize(w*h*c), BLOCK>>>(im_data, w, h, c);
		check_error(cudaPeekAtLastError());
	}
	if (vflip) {
		flip_image_vertical_kernel<<<cuda_gridsize(w*h*c), BLOCK>>>(im_data, w, h, c);
		check_error(cudaPeekAtLastError());
	}
}

__global__ void cutout_image_kernel(float* im_data, int w, int h, int c, int cut_xs, int cut_xe, int cut_ys, int cut_ye) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= w*h*c) return;
	int i = index % w;
	index = (index - i) / w;
	int j = index % h;
	int k = (index - j) / h;
	if (i < cut_xs || i > cut_xe || j < cut_ys || j > cut_ye) return;
	im_data[k*h*w + j*w + i] = 0.5;
}

void cutout_image_gpu(float* im_data, int w, int h, int c, int cut_xs, int cut_xe, int cut_ys, int cut_ye) {
	cutout_image_kernel<<<cuda_gridsize(w*h*c), BLOCK>>>(im_data, w, h, c, cut_xs, cut_xe, cut_ys, cut_ye);
	check_error(cudaPeekAtLastError());
}

void random_cutout_image_gpu(float* im_data, int w, int h, int c, cutout_args cutout) {
	if ((rand() % 100) < (int)(cutout.prob * 100.)) {

        int xc = rand() % w;
        int yc = rand() % h;
        
        int cut_w = (rand() % cutout.max_w) + 1;
        int cut_h = (rand() % cutout.max_h) + 1;
        
        int cut_xs = xc - cut_w/2;
        int cut_xe = cut_xs + cut_w - 1;

        cut_xs = cut_xs > 0 ? cut_xs : 0;
        cut_xe = cut_xe < w ? cut_xe : w - 1;

        int cut_ys = yc - cut_h/2;
        int cut_ye = cut_ys + cut_h - 1;
        
        cut_ys = cut_ys > 0 ? cut_ys : 0;
        cut_ye = cut_ye < h ? cut_ye : h - 1;

        cutout_image_gpu(im_data, w, h, c, cut_xs, cut_xe, cut_ys, cut_ye);
    }
}

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

__global__ void scale_image_channel_kernel(float* im_data, int w, int h, int k, float v) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= w*h) return;
	im_data[k*h*w + index] *= v;
}

void scale_image_channel_gpu(float* im_data, int w, int h, int c, int k, float v) {
	if ((k >= 0) && (k < c)) {
		scale_image_channel_kernel<<<cuda_gridsize(h*w), BLOCK>>>(im_data, w, h, k, v);
		check_error(cudaPeekAtLastError());
	}
}

__global__ void add_hue_constrain_image_channel_kernel(float* im_data, int w, int h, int k, float v) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= w*h) return;
	im_data[k*h*w + index] += v;
	if (im_data[k*h*w + index] > 1) im_data[k*h*w + index] -= 1;
	if (im_data[k*h*w + index] < 0) im_data[k*h*w + index] += 1;
}

void add_hue_constrain_image_channel_gpu(float* im_data, int w, int h, int c, int k, float v) {
	if ((k >= 0) && (k < c)) {
		add_hue_constrain_image_channel_kernel<<<cuda_gridsize(h*w), BLOCK>>>(im_data, w, h, k, v);
		check_error(cudaPeekAtLastError());
	}
}

__global__ void constrain_image_kernel(float* im_data, int w, int h, int c) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= c*h*w) return;
	if (im_data[index] < 0) im_data[index] = 0;
	if (im_data[index] > 1) im_data[index] = 1;
}

void constrain_image_gpu(float* im_data, int w, int h, int c) {
	constrain_image_kernel<<<cuda_gridsize(c*h*w), BLOCK>>>(im_data, w, h, c);
	check_error(cudaPeekAtLastError());
}

void distort_image_gpu(float* im_data, int w, int h, int c, float hue, float sat, float val) {
	rgb_to_hsv_gpu(im_data, w, h, c);
	scale_image_channel_gpu(im_data, w, h, c, 1, sat);
	scale_image_channel_gpu(im_data, w, h, c, 2, val);
	add_hue_constrain_image_channel_gpu(im_data, w, h, c, 0, hue);
	hsv_to_rgb_gpu(im_data, w, h, c);
	constrain_image_gpu(im_data, w, h, c);
}

void random_distort_image_gpu(float* im_data, int w, int h, int c, float hue, float saturation, float exposure) {
    float dhue = rand_uniform(-hue, hue);
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);
    distort_image_gpu(im_data, w, h, c, dhue, dsat, dexp);
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

__global__ void add_image_noise_kernel(float* im_data, int w, int h, int c, float noise) {
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (index >= c*h*w) return;
	im_data[index] += noise;
	if (im_data[index] < 0) im_data[index] = 0;
	if (im_data[index] > 1) im_data[index] = 1;
}

void add_image_noise_gpu(float* im_data, int w, int h, int c, float noise) {
	add_image_noise_kernel<<<cuda_gridsize(c*h*w), BLOCK>>>(im_data, w, h, c, noise);
	check_error(cudaPeekAtLastError());
}

void random_distort_image_extend_gpu(float* im_data, int w, int h, int c, float solarize, float posterize, float noise) {
	float psolarize = rand_uniform(0, 1);
	if (psolarize < solarize) {
		float dsolarize = rand_uniform(0, 1);
		solarize_image_gpu(im_data, w, h, c, dsolarize);
	}

	float pposterize = rand_uniform(0, 1);
	if (pposterize < posterize) {
		int dposterize = rand_int(16, 256);
		posterize_image_gpu(im_data, w, h, c, dposterize);
	}
	int pnoise = rand() % 2;
	if (noise > 0 && pnoise) {
		noise = (rand_uniform(0, 1) * 2 - 1) * noise;
		add_image_noise_gpu(im_data, w, h, c, noise);
	}
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