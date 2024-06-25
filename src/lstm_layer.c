#include "lstm_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"
#include "initializer.h"

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

void init_lstm_layer(layer l, initializer init) {
    init_connected_layer(*(l.wi), l.initializer);
    init_connected_layer(*(l.wf), l.initializer);
    init_connected_layer(*(l.wo), l.initializer);
    init_connected_layer(*(l.wg), l.initializer);
    init_connected_layer(*(l.ui), l.initializer);
    init_connected_layer(*(l.uf), l.initializer);
    init_connected_layer(*(l.uo), l.initializer);
    init_connected_layer(*(l.ug), l.initializer);
}

layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize, OPTIMIZER optim)
{
    fprintf(stderr, "LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = { 0 };
    l.batch = batch;
    l.type = LSTM;
    l.steps = steps;
    l.inputs = inputs;

    l.uf = malloc(sizeof(layer));
    *(l.uf) = make_connected_layer(batch*steps, inputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.uf->batch = batch;

    l.ui = malloc(sizeof(layer));
    *(l.ui) = make_connected_layer(batch*steps, inputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.ui->batch = batch;

    l.ug = malloc(sizeof(layer));
    *(l.ug) = make_connected_layer(batch*steps, inputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.ug->batch = batch;

    l.uo = malloc(sizeof(layer));
    *(l.uo) = make_connected_layer(batch*steps, inputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.uo->batch = batch;

    l.wf = malloc(sizeof(layer));
    *(l.wf) = make_connected_layer(batch*steps, outputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.wf->batch = batch;

    l.wi = malloc(sizeof(layer));
    *(l.wi) = make_connected_layer(batch*steps, outputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.wi->batch = batch;

    l.wg = malloc(sizeof(layer));
    *(l.wg) = make_connected_layer(batch*steps, outputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.wg->batch = batch;

    l.wo = malloc(sizeof(layer));
    *(l.wo) = make_connected_layer(batch*steps, outputs, outputs, (activation_scheme){LINEAR}, batch_normalize, optim);
    l.wo->batch = batch;

    l.batch_normalize = batch_normalize;
    l.outputs = outputs;

    l.output = calloc(outputs*batch*steps, sizeof(float));
    l.state = calloc(outputs*batch, sizeof(float));

    l.prev_state_cpu =  calloc(batch*outputs, sizeof(float));
    l.prev_cell_cpu =   calloc(batch*outputs, sizeof(float));
    l.cell_cpu =        calloc(batch*outputs*steps, sizeof(float));

    l.f_cpu =           calloc(batch*outputs, sizeof(float));
    l.i_cpu =           calloc(batch*outputs, sizeof(float));
    l.g_cpu =           calloc(batch*outputs, sizeof(float));
    l.o_cpu =           calloc(batch*outputs, sizeof(float));
    l.c_cpu =           calloc(batch*outputs, sizeof(float));
    l.h_cpu =           calloc(batch*outputs, sizeof(float));
    l.temp_cpu =        calloc(batch*outputs, sizeof(float));
    l.temp2_cpu =       calloc(batch*outputs, sizeof(float));
    l.temp3_cpu =       calloc(batch*outputs, sizeof(float));
    l.dc_cpu =          calloc(batch*outputs, sizeof(float));
    l.dh_cpu =          calloc(batch*outputs, sizeof(float));

    l.forward_gpu = forward_lstm_layer_gpu;
    l.backward_gpu = backward_lstm_layer_gpu;

    l.update_gpu = update_lstm_layer_gpu;

    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps);

    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.prev_cell_gpu = cuda_make_array(0, batch*outputs);
    l.cell_gpu = cuda_make_array(0, batch*outputs*steps);

    l.f_gpu = cuda_make_array(0, batch*outputs);
    l.i_gpu = cuda_make_array(0, batch*outputs);
    l.g_gpu = cuda_make_array(0, batch*outputs);
    l.o_gpu = cuda_make_array(0, batch*outputs);
    l.c_gpu = cuda_make_array(0, batch*outputs);
    l.h_gpu = cuda_make_array(0, batch*outputs);
    l.temp_gpu =  cuda_make_array(0, batch*outputs);
    l.temp2_gpu = cuda_make_array(0, batch*outputs);
    l.temp3_gpu = cuda_make_array(0, batch*outputs);
    l.dc_gpu = cuda_make_array(0, batch*outputs);
    l.dh_gpu = cuda_make_array(0, batch*outputs);

    cudnnSetTensor4dDescriptor(l.wf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wf->out_c, l.wf->out_h, l.wf->out_w); 
    cudnnSetTensor4dDescriptor(l.wi->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wi->out_c, l.wi->out_h, l.wi->out_w); 
    cudnnSetTensor4dDescriptor(l.wg->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wg->out_c, l.wg->out_h, l.wg->out_w); 
    cudnnSetTensor4dDescriptor(l.wo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.wo->out_c, l.wo->out_h, l.wo->out_w); 

    cudnnSetTensor4dDescriptor(l.uf->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uf->out_c, l.uf->out_h, l.uf->out_w); 
    cudnnSetTensor4dDescriptor(l.ui->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ui->out_c, l.ui->out_h, l.ui->out_w); 
    cudnnSetTensor4dDescriptor(l.ug->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.ug->out_c, l.ug->out_h, l.ug->out_w); 
    cudnnSetTensor4dDescriptor(l.uo->dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, l.uo->out_c, l.uo->out_h, l.uo->out_w); 

    return l;
}

void update_lstm_layer_gpu(layer l, update_args a)
{
    update_connected_layer_gpu(*(l.wf), a);
    update_connected_layer_gpu(*(l.wi), a);
    update_connected_layer_gpu(*(l.wg), a);
    update_connected_layer_gpu(*(l.wo), a);
    update_connected_layer_gpu(*(l.uf), a);
    update_connected_layer_gpu(*(l.ui), a);
    update_connected_layer_gpu(*(l.ug), a);
    update_connected_layer_gpu(*(l.uo), a);
}

void forward_lstm_layer_gpu(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);

    fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
    if (state.train) {
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input_gpu = l.h_gpu;
        forward_connected_layer_gpu(wf, s);							
        forward_connected_layer_gpu(wi, s);							
        forward_connected_layer_gpu(wg, s);							
        forward_connected_layer_gpu(wo, s);							

        s.input_gpu = state.input_gpu;
        forward_connected_layer_gpu(uf, s);							
        forward_connected_layer_gpu(ui, s);							
        forward_connected_layer_gpu(ug, s);							
        forward_connected_layer_gpu(uo, s);							

        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);	

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);	
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});		
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});		
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, (activation_scheme){TANH});			
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});		

        copy_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);			
        activate_array_gpu(l.h_gpu, l.outputs*l.batch, (activation_scheme){TANH});		
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);	

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);		
        copy_gpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1);

        state.input_gpu += l.inputs*l.batch;
        l.output_gpu    += l.outputs*l.batch;
        l.cell_gpu      += l.outputs*l.batch;

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}

void backward_lstm_layer_gpu(layer l, network state)
{
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    uf.impact = l.impact;
    ui.impact = l.impact;
    ug.impact = l.impact;
    uo.impact = l.impact;

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input_gpu += l.inputs*l.batch*(l.steps - 1);
    if (state.delta_gpu) state.delta_gpu += l.inputs*l.batch*(l.steps - 1);

    l.output_gpu += l.outputs*l.batch*(l.steps - 1);
    l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
    l.delta_gpu += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_gpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
        if (i != 0) copy_gpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);

        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_gpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);
        axpy_gpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);			

        copy_gpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);			
        axpy_gpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);

        activate_array_gpu(l.f_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});			
        activate_array_gpu(l.i_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});		
        activate_array_gpu(l.g_gpu, l.outputs*l.batch, (activation_scheme){TANH});			
        activate_array_gpu(l.o_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC});		

        copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);		

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, (activation_scheme){TANH});			

        copy_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);
        mul_gpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);	

        gradient_array_gpu(l.temp_gpu, l.outputs*l.batch, (activation_scheme){TANH}, l.temp2_gpu);
        axpy_gpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);		

        copy_gpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			
        activate_array_gpu(l.temp_gpu, l.outputs*l.batch, (activation_scheme){TANH});			
        mul_gpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);		
        gradient_array_gpu(l.o_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC}, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;															
        backward_connected_layer_gpu(wo, s);	

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uo, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);				
        gradient_array_gpu(l.g_gpu, l.outputs*l.batch, (activation_scheme){TANH}, l.temp_gpu);		
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;														
        backward_connected_layer_gpu(wg, s);	

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ug, s);																

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);				
        gradient_array_gpu(l.i_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC}, l.temp_gpu);	
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wi, s);						

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(ui, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);		
        mul_gpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.f_gpu, l.outputs*l.batch, (activation_scheme){LOGISTIC}, l.temp_gpu);
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
        s.input_gpu = l.prev_state_gpu;
        s.delta_gpu = l.dh_gpu;
        backward_connected_layer_gpu(wf, s);						

        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
        s.input_gpu = state.input_gpu;
        s.delta_gpu = state.delta_gpu;
        backward_connected_layer_gpu(uf, s);									

        copy_gpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			
        mul_gpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);				
        copy_gpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);				

        state.input_gpu -= l.inputs*l.batch;
        if (state.delta_gpu) state.delta_gpu -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.cell_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}