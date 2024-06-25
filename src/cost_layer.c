#include "cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "seg")==0) return SEG;
    if (strcmp(s, "sse")==0 || strcmp(s, "l2")==0) return SSE;
    if (strcmp(s, "masked")==0) return MASKED;
    if (strcmp(s, "smooth")==0) return SMOOTH;
    if (strcmp(s, "l1")==0) return L1;
    if (strcmp(s, "wgan")==0) return WGAN;
    if (strcmp(s, "symexp")==0) return SYMEXP;
    if (strcmp(s, "logcosh")==0) return LOGCOSH;
    fprintf(stderr, "Couldn't find cost type %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SEG:
            return "seg";
        case SSE:
            return "sse";
        case MASKED:
            return "masked";
        case SMOOTH:
            return "smooth";
        case L1:
            return "l1";
        case WGAN:
            return "wgan";
        case SYMEXP:
            return "symexp";
        case LOGCOSH:
            return "logcosh";
    }
    return "sse";
}

layer make_cost_layer(int batch, int inputs, COST_TYPE cost_type, float scale)
{
    layer l = {0};
    l.type = COST;

    l.scale = scale;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.cost_type = cost_type;
    l.delta = calloc(inputs*batch, sizeof(float));
    l.output = calloc(inputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));

    l.forward_gpu = forward_cost_layer_gpu;
    l.backward_gpu = backward_cost_layer_gpu;

    l.delta_gpu = cuda_make_array(l.output, inputs*batch);
    l.output_gpu = cuda_make_array(l.delta, inputs*batch);
    return l;
}

void resize_cost_layer(layer *l, int inputs)
{
    l->inputs = inputs;
    l->outputs = inputs;
    l->delta = realloc(l->delta, inputs*l->batch*sizeof(float));
    l->output = realloc(l->output, inputs*l->batch*sizeof(float));

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    l->delta_gpu = cuda_make_array(l->delta, inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, inputs*l->batch);
}

int float_abs_compare (const void * a, const void * b)
{
    float fa = *(const float*) a;
    if(fa < 0) fa = -fa;
    float fb = *(const float*) b;
    if(fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

void forward_cost_layer_gpu(layer l, network net)
{
    if (!net.truth) return;
    if(l.smooth){
        scal_gpu(l.batch*l.inputs, (1-l.smooth), net.truth_gpu, 1);
        add_gpu(l.batch*l.inputs, l.smooth * 1./l.inputs, net.truth_gpu, 1);
    }

    if(l.cost_type == SMOOTH){
        smooth_l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == L1){
        l1_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == WGAN){
        wgan_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == SYMEXP){
        symexp_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else if (l.cost_type == LOGCOSH){
        logcosh_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    } else {
        l2_gpu(l.batch*l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    }

    if (l.cost_type == SEG && l.noobject_scale != 1) {
        scale_mask_gpu(l.batch*l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
        scale_mask_gpu(l.batch*l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale);
    }
    if (l.cost_type == MASKED) {
        mask_gpu(l.batch*l.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
    }

    if(l.ratio){
        cuda_pull_array(l.delta_gpu, l.delta, l.batch*l.inputs);
        qsort(l.delta, l.batch*l.inputs, sizeof(float), float_abs_compare);
        int n = (1-l.ratio) * l.batch*l.inputs;
        float thresh = l.delta[n];
        thresh = 0;
        printf("%f\n", thresh);
        supp_gpu(l.batch*l.inputs, thresh, l.delta_gpu, 1);
    }

    if(l.thresh){
        supp_gpu(l.batch*l.inputs, l.thresh*1./l.inputs, l.delta_gpu, 1);
    }

    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    l.cost[0] = sum_array(l.output, l.batch*l.inputs);
}

void backward_cost_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.scale * l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}