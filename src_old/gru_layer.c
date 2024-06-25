#include "gru_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
}

void init_gru_layer(layer l, initializer init) {
    init_connected_layer(*(l.wz), l.initializer);
    init_connected_layer(*(l.wr), l.initializer);
    init_connected_layer(*(l.wh), l.initializer);
    init_connected_layer(*(l.uz), l.initializer);
    init_connected_layer(*(l.ur), l.initializer);
    init_connected_layer(*(l.uh), l.initializer);
}

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, OPTIMIZER optim)
{
    batch = batch / steps;
    layer l = {0};
    l.batch = batch;
    l.type = GRU;
    l.steps = steps;
    l.inputs = inputs;

    l.uz = malloc(sizeof(layer));
    *(l.uz) = make_connected_layer(batch*steps, inputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.uz->batch = batch;

    l.wz = malloc(sizeof(layer));
    *(l.wz) = make_connected_layer(batch*steps, outputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.wz->batch = batch;

    l.ur = malloc(sizeof(layer));
    *(l.ur) = make_connected_layer(batch*steps, inputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.ur->batch = batch;

    l.wr = malloc(sizeof(layer));
    *(l.wr) = make_connected_layer(batch*steps, outputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.wr->batch = batch;

    l.uh = malloc(sizeof(layer));
    *(l.uh) = make_connected_layer(batch*steps, inputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.uh->batch = batch;

    l.wh = malloc(sizeof(layer));
    *(l.wh) = make_connected_layer(batch*steps, outputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.wh->batch = batch;

    l.batch_normalize = batch_normalize;

    l.outputs = outputs;
    l.output = calloc(outputs*batch*steps, sizeof(float));
    l.delta = calloc(outputs*batch*steps, sizeof(float));
    l.state = calloc(outputs*batch, sizeof(float));
    l.prev_state = calloc(outputs*batch, sizeof(float));
    l.forgot_state = calloc(outputs*batch, sizeof(float));
    l.forgot_delta = calloc(outputs*batch, sizeof(float));

    l.r_cpu = calloc(outputs*batch, sizeof(float));
    l.z_cpu = calloc(outputs*batch, sizeof(float));
    l.h_cpu = calloc(outputs*batch, sizeof(float));

    l.forward_gpu = forward_gru_layer_gpu;
    l.backward_gpu = backward_gru_layer_gpu;
    
    l.update_gpu = update_gru_layer_gpu;

    l.forgot_state_gpu = cuda_make_array(0, batch*outputs);
    l.forgot_delta_gpu = cuda_make_array(0, batch*outputs);
    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.state_gpu = cuda_make_array(0, batch*outputs);
    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(0, batch*outputs*steps);
    l.r_gpu = cuda_make_array(0, batch*outputs);
    l.z_gpu = cuda_make_array(0, batch*outputs);
    l.h_gpu = cuda_make_array(0, batch*outputs);

    cudnnSetTensor4dDescriptor(l.uz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uz->out_c, l.uz->out_h, l.uz->out_w); 
    cudnnSetTensor4dDescriptor(l.uh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uh->out_c, l.uh->out_h, l.uh->out_w); 
    cudnnSetTensor4dDescriptor(l.ur->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ur->out_c, l.ur->out_h, l.ur->out_w); 
    cudnnSetTensor4dDescriptor(l.wz->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wz->out_c, l.wz->out_h, l.wz->out_w); 
    cudnnSetTensor4dDescriptor(l.wh->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wh->out_c, l.wh->out_h, l.wh->out_w); 
    cudnnSetTensor4dDescriptor(l.wr->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wr->out_c, l.wr->out_h, l.wr->out_w); 

    return l;
}

void pull_gru_layer(layer l)
{
}

void push_gru_layer(layer l)
{
}

void update_gru_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.ur), a);
    update_connected_layer_gpu(*(l.uz), a);
    update_connected_layer_gpu(*(l.uh), a);
    update_connected_layer_gpu(*(l.wr), a);
    update_connected_layer_gpu(*(l.wz), a);
    update_connected_layer_gpu(*(l.wh), a);
}

void forward_gru_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    fill_gpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);

    fill_gpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);
    if(net.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = l.state_gpu;
        forward_connected_layer_gpu(wz, s);
        forward_connected_layer_gpu(wr, s);

        s.input_gpu = net.input_gpu;
        forward_connected_layer_gpu(uz, s);
        forward_connected_layer_gpu(ur, s);
        forward_connected_layer_gpu(uh, s);

        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

        activate_array_gpu(l.z_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});
        activate_array_gpu(l.r_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});

        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);

        s.input_gpu = l.forgot_state_gpu;
        forward_connected_layer_gpu(wh, s);

        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);


        if(l.tanh){
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, (activation_scheme){TANH});
        } else {
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});
        }

        weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.state_gpu, 1);

        net.input_gpu += l.inputs*l.batch;
        l.output_gpu += l.outputs*l.batch;
        increment_layer(&uz, 1);
        increment_layer(&ur, 1);
        increment_layer(&uh, 1);

        increment_layer(&wz, 1);
        increment_layer(&wr, 1);
        increment_layer(&wh, 1);
    }
}

void backward_gru_layer_gpu(layer l, network net)
{
    network s = {0};
    s.train = net.train;
    int i;
    layer uz = *(l.uz);
    layer ur = *(l.ur);
    layer uh = *(l.uh);

    uz.impact = l.impact;
    ur.impact = l.impact;
    uh.impact = l.impact;

    layer wz = *(l.wz);
    layer wr = *(l.wr);
    layer wh = *(l.wh);

    increment_layer(&uz, l.steps - 1);
    increment_layer(&ur, l.steps - 1);
    increment_layer(&uh, l.steps - 1);

    increment_layer(&wz, l.steps - 1);
    increment_layer(&wr, l.steps - 1);
    increment_layer(&wh, l.steps - 1);

    net.input_gpu += l.inputs*l.batch*(l.steps-1);
    if(net.delta_gpu) net.delta_gpu += l.inputs*l.batch*(l.steps-1);
    l.output_gpu += l.outputs*l.batch*(l.steps-1);
    l.delta_gpu += l.outputs*l.batch*(l.steps-1);
    float *end_state = l.output_gpu;
    for (i = l.steps-1; i >= 0; --i) {
        if(i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.state_gpu, 1);
        else copy_gpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        float *prev_delta_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_gpu(l.outputs*l.batch, uz.output_gpu, 1, l.z_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);

        copy_gpu(l.outputs*l.batch, ur.output_gpu, 1, l.r_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);

        activate_array_gpu(l.z_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});
        activate_array_gpu(l.r_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});

        copy_gpu(l.outputs*l.batch, uh.output_gpu, 1, l.h_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);

        if(l.tanh){
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, (activation_scheme){TANH});
        } else {
            activate_array_gpu(l.h_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});
        }

        weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs*l.batch, l.delta_gpu);

        if(l.tanh){
            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, (activation_scheme){TANH}, uh.delta_gpu);
        } else {
            gradient_array_gpu(l.h_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC}, uh.delta_gpu);
        }

        copy_gpu(l.outputs*l.batch, uh.delta_gpu, 1, wh.delta_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
        fill_gpu(l.outputs*l.batch, 0, l.forgot_delta_gpu, 1);

        s.input_gpu = l.forgot_state_gpu;
        s.delta_gpu = l.forgot_delta_gpu;

        backward_connected_layer_gpu(wh, s);
        if(prev_delta_gpu) mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
        mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.state_gpu, ur.delta_gpu);

        gradient_array_gpu(l.r_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC}, ur.delta_gpu);
        copy_gpu(l.outputs*l.batch, ur.delta_gpu, 1, wr.delta_gpu, 1);

        gradient_array_gpu(l.z_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC}, uz.delta_gpu);
        copy_gpu(l.outputs*l.batch, uz.delta_gpu, 1, wz.delta_gpu, 1);

        s.input_gpu = l.state_gpu;
        s.delta_gpu = prev_delta_gpu;

        backward_connected_layer_gpu(wr, s);
        backward_connected_layer_gpu(wz, s);

        s.input_gpu = net.input_gpu;
        s.delta_gpu = net.delta_gpu;

        backward_connected_layer_gpu(uh, s);
        backward_connected_layer_gpu(ur, s);
        backward_connected_layer_gpu(uz, s);


        net.input_gpu -= l.inputs*l.batch;
        if(net.delta_gpu) net.delta_gpu -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;
        increment_layer(&uz, -1);
        increment_layer(&ur, -1);
        increment_layer(&uh, -1);

        increment_layer(&wz, -1);
        increment_layer(&wr, -1);
        increment_layer(&wh, -1);
    }
    copy_gpu(l.outputs*l.batch, end_state, 1, l.state_gpu, 1);
}