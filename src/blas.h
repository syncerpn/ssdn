#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"
#include "cuda.h"
#include "tree.h"

void flatten(float *x, int size, int layers, int batch, int forward);
void constrain_gpu(int N, float ALPHA, float * X, int INCX);
void softmax(float *input, int n, float temp, int stride, float *output);
void stretch_fill_3d_gpu(float* x, float* y, int w, int pad_w, int h, int pad_h, int c, int pad_c);
void squeeze_fill_3d_gpu(float* x, float* y, int w, int pad_w, int h, int pad_h, int c, int pad_c);
void deconv_transpose_weights(float* weights, int spatial_size, int channels);
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void copy_gpu_offset(int N, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, float * X, int INCX);
void supp_gpu(int N, float ALPHA, float * X, int INCX);
void mask_gpu(int N, float * X, float mask_num, float * mask, float val);
void scale_mask_gpu(int N, float * X, float mask_num, float * mask, float scale);
void const_gpu(int N, float ALPHA, float *X, int INCX);
void pow_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_gpu(int N, float *X, int INCX, float *Y, int INCY);
void mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void l2normalize_gpu(float *x, float *dx, int batch, int filters, int spatial);
void normalize_delta_gpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void fast_mean_delta_gpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void fast_variance_delta_gpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_scale_gpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void backward_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
void logistic_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_gpu(int n, float *pred, float *truth, float *delta, float *error);
void smooth_l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_gpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_gpu(int n, float *pred, float *truth, float *delta, float *error);
void symexp_gpu(int n, float *pred, float *truth, float *delta, float *error); //nghiant_20190822: symmetric exp loss
void logcosh_gpu(int n, float *pred, float *truth, float *delta, float *error); //nghiant_20190826: log cosh loss
void wgan_gpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_delta_gpu(float *a, float *b, float *s, float *da, float *db, float *ds, int num, float *dc);
void weighted_sum_gpu(float *a, float *b, float *s, int num, float *c);
void mult_add_into_gpu(int num, float *a, float *b, float *c);
void inter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_gpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void reorg_gpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out, float ALPHA, float BETA);
void softmax_gpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void adam_update_gpu(float *w, float *d, float *m, float *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, float *x, float *m, float *v, float B1, float B2, float rate, float eps, int t);
void flatten_gpu(float *x, int spatial, int layers, int batch, int forward, float *out, float ALPHA, float BETA);
void softmax_tree(float *input, int spatial, int batch, int stride, float temp, float *output, tree hier);
void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
void stack_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
void sr_flat_gpu(float *in, int w, int h, int c, int u, int batch, int forward, float scale, float *out);
void gradient_clipping_gpu(float* x, float N, float clip, float norm);
//nghiant: norm_w
void norm_w_normalize_gpu(float* v, float* v_norm, int size, int n);
//combined method and 2-step separate method are equivalent
void norm_w_reform_weights_gpu(float* weights, float* v, float* v_norm, float* g, int size, int n); //combined method
void norm_w_norm_div_weights_gpu(float* weights, float* v, float* v_norm, int size, int n); //separate method: step 1
void norm_w_scale_g_weights_gpu(float* weights, float* g, int size, int n); //separate method: step 2
void norm_w_delta_g_gpu(float* g_delta, float* weights_delta, float* v, float* v_norm, int size, int n);
void norm_w_delta_v_gpu(float* v_delta, float* weights_delta, float* g_delta, float* v, float* v_norm, float* g, int size, int n);
//nghiant_end
//nghiant:sam layer
void sam_gpu(float *in_w_h_c, int size, int channel_size, float *scales_c, float *out);
void backward_sam_gpu(float *in_w_h_c_delta, int size, int channel_size, float *in_scales_c, float *out_from_delta, float *in_from_output, float *out_state_delta);
//nghiant_end
void scale_channels_gpu(float *in_w_h_c, int size, int channel_size, int batch_size, int scale_wh, float *scales_c, float *out);
void backward_scale_channels_gpu(float *in_w_h_c_delta, int size, int channel_size, int batch_size, int scale_wh, float *in_scales_c, float *out_from_delta, float *in_from_output, float *out_state_delta);
#endif