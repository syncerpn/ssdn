#include "quantization_layer.h"
#include <stdio.h>


char* get_quantization_scheme_string(QUANTIZATION_SCHEME type) {
    switch(type){
        case QS_UNIFORM:
            return "UNIFORM";
        case QS_ROOT:
            return "ROOT";
        case QS_BIT:
            return "BIT";
        case QS_NONE:
            return "NONE";
        default:
            return "NONE";
    }
}

void print_quantization_scheme_summary(quantization_scheme qs) {
    char buff[64];
    int arg_1 = 5;
    int arg_2 = 5;
    int arg_2d= 3;
    int arg_3 = 5;
    int arg_3d= 3;
    switch(qs.type){
        case QS_UNIFORM:
            sprintf(buff, "%s %*d %*.*f %*s-", get_quantization_scheme_string(qs.type), arg_1, qs.num_level, arg_2, arg_2d, qs.step_size, arg_3-1,"");
            break;
        case QS_ROOT:
            sprintf(buff, "%s %*d %*.*f %*.*f", get_quantization_scheme_string(qs.type), arg_1, qs.num_level, arg_2, arg_2d, qs.step_size, arg_3, arg_3d, qs.root);
            break;
        case QS_BIT:
            sprintf(buff, "%s %*d %*.*f %*s-", get_quantization_scheme_string(qs.type), arg_1, qs.num_level, arg_2, arg_2d, qs.step_size, arg_3-1,"");
            break;
        case QS_NONE:
        default:
            sprintf(buff, "%s %*s- %*s- %*s-", get_quantization_scheme_string(qs.type), arg_1-1, "", arg_2-1, "", arg_3-1, "");
            break;
    }
    fprintf(stderr, " %26s ", buff);
    return;
}

QUANTIZATION_SCHEME get_quantization_scheme(char *s) {
    if (strcmp(s, "qs_uniform")==0) return QS_UNIFORM;
    if (strcmp(s, "qs_root")==0) return QS_ROOT;
    if (strcmp(s, "qs_bit")==0) return QS_BIT;
    if (strcmp(s, "qs_none")==0) return QS_NONE;
    fprintf(stderr, "Couldn't find quantization scheme %s, going with QS_NONE\n", s);
    return QS_NONE;
}

layer make_quantization_layer(int batch, int w, int h, int c, quantization_scheme qs)
{
    layer l = {0};
    l.type = QUANTIZATION;
    l.batch = batch;


    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;

    l.inputs = l.h * l.w * l.c;
    l.outputs = l.out_h * l.out_w * l.out_c;

    l.output = calloc(l.inputs * batch, sizeof(float));
    l.delta  = calloc(l.inputs * batch, sizeof(float));

    l.forward_gpu = forward_quantization_layer_gpu;
    l.backward_gpu = backward_quantization_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, l.inputs * batch);
    l.delta_gpu = cuda_make_array(l.delta, l.inputs * batch);

    l.quantization = qs;

    return l;
}

void resize_quantization_layer(layer* l, int w, int h) {
    l->w = w;
    l->h = h;

    l->out_w = w;
    l->out_h = h;

    l->inputs = w * h * l->c;
    l->outputs = l->inputs;

    l->output = realloc(l->output, l->inputs * l->batch * sizeof(float));
    l->delta  = realloc(l->delta , l->inputs * l->batch * sizeof(float));

    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);

    l->output_gpu = cuda_make_array(l->output, l->inputs * l->batch);
    l->delta_gpu  = cuda_make_array(l->delta , l->inputs * l->batch);
}