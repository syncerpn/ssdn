#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"
#include "quantization_layer.h"
#include "diff_layer.h"
#include "stack_layer.h"
#include "sr_flat_layer.h"
#include "norm_w_convolutional_layer.h"
#include "sam_layer.h"
#include "scale_channels_layer.h"
#include "customizable_conv.h"
#include "initializer.h"

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]")==0) return YOLO;
    if (strcmp(type, "[iseg]")==0) return ISEG;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[deconv]")==0
            || strcmp(type, "[deconvolutional]")==0) return DECONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[logistic]")==0) return LOGXENT;
    if (strcmp(type, "[l2norm]")==0) return L2NORM;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]") == 0) return LSTM;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[reorg]")==0) return REORG;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE;
    if (strcmp(type, "[quantization]")==0) return QUANTIZATION;
    if (strcmp(type, "[diff]")==0) return DIFF;
    if (strcmp(type, "[stack]")==0) return STACK;
    if (strcmp(type, "[sr_flat]")==0) return SR_FLAT;
    if (strcmp(type, "[norm_w_conv]")==0
            || strcmp(type, "[norm_w_convolutional]")==0) return NORM_W_CONVOLUTIONAL;
    if (strcmp(type, "[sam]")==0) return SAM;
    if (strcmp(type, "[scale_channels]")==0) return SCALE_CHANNELS;
    if (strcmp(type, "[custom_conv]")==0) return CUSTOM_CONV;
    return BLANK;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
    int index;
    int time_steps;
    network *net;
} size_params;

layer parse_local(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);

    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before local layer must output image.");

    layer l = make_local_layer(batch,h,w,c,n,size,stride,pad,activation,params.net->optim);

    return l;
}

layer parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);


    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    layer l = make_deconvolutional_layer(batch,h,w,c,n,size,stride,padding, activation, batch_normalize, params.net->optim);

    return l;
}

layer parse_convolutional(list *options, size_params params)
{
    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");

    int n, groups;

    //nghiant: depth-wise separable convolution; override other settings
    int depthwise = option_find_int_quiet(options, "depthwise", 0);
    if (depthwise) {
        n = c;
        groups = c;
    } else {
        n = option_find_int(options, "filters",1);
        groups = option_find_int_quiet(options, "groups", 1);
    }

    //nghiant: add dilation option
    int dilation = option_find_int_quiet(options, "dilation", 1);
    
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);

    //nghiant: new padding with dilation below
    if (pad) padding = (size + (dilation - 1) * (size - 1)) /2;
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);


    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    
    //nghiant
    char *quantization_s = option_find_str_quiet(options, "quantization_scheme", "qs_none");
    quantization_scheme qs = {0};
    qs.type = get_quantization_scheme(quantization_s);
    qs.num_level = option_find_int_quiet(options, "qs_num_level", 64);
    qs.step_size = option_find_float_quiet(options, "qs_step_size", 0.125);
    qs.root = option_find_float_quiet(options, "qs_root", 0.0);
    qs.zero_center = option_find_int_quiet(options, "qs_zero_center", 0);

    char* type = option_find_str_quiet(options, "weight_transform_scheme", "wts_none");
    weight_transform_scheme wts = {0};
    wts.type = get_weight_transform_scheme(type);
    wts.step_size = option_find_float_quiet(options, "wts_step_size", 0.125);
    wts.num_level = option_find_int_quiet(options, "wts_num_level", 64);
    wts.first_shifting_factor = option_find_int_quiet(options, "wts_first_shifting_factor", 0);

    layer l = make_convolutional_layer(batch,h,w,c,n,dilation,groups,size,stride,padding,activation, batch_normalize, params.net->optim);

    assign_weight_transform_convolutional_layer(&l, wts);
    assign_quantization_convolutional_layer(&l, qs);

    l.flipped = option_find_int_quiet(options, "flipped", 0);

    return l;
}

layer parse_custom_conv(list *options, size_params params)
{
    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before customizable convolutional layer must output image.");

    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);

    //nghiant: new padding with dilation below
    if (pad) padding = size / 2;
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);
    
    layer l = make_custom_conv_layer(batch, h, w, c, n, size, stride, padding, activation);

    return l;
}

layer parse_norm_w_convolutional(list *options, size_params params)
{
    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");

    int n, groups;

    //nghiant: depth-wise separable convolution; override other settings
    int depthwise = option_find_int_quiet(options, "depthwise", 0);
    if (depthwise) {
        n = c;
        groups = c;
    } else {
        n = option_find_int(options, "filters",1);
        groups = option_find_int_quiet(options, "groups", 1);
    }

    //nghiant: add dilation option
    int dilation = option_find_int_quiet(options, "dilation", 1);
    
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);

    //nghiant: new padding with dilation below
    if (pad) padding = (size + (dilation - 1) * (size - 1)) /2;
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);


    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    
    //nghiant
    char *quantization_s = option_find_str_quiet(options, "quantization_scheme", "qs_none");
    quantization_scheme qs = {0};
    qs.type = get_quantization_scheme(quantization_s);
    qs.num_level = option_find_int_quiet(options, "qs_num_level", 64);
    qs.step_size = option_find_float_quiet(options, "qs_step_size", 0.125);
    qs.root = option_find_float_quiet(options, "qs_root", 0.0);
    qs.zero_center = option_find_int_quiet(options, "qs_zero_center", 0);

    char* type = option_find_str_quiet(options, "weight_transform_scheme", "wts_none");
    weight_transform_scheme wts = {0};
    wts.type = get_weight_transform_scheme(type);
    wts.step_size = option_find_float_quiet(options, "wts_step_size", 0.125);
    wts.num_level = option_find_int_quiet(options, "wts_num_level", 64);
    wts.first_shifting_factor = option_find_int_quiet(options, "wts_first_shifting_factor", 0);

    layer l = make_norm_w_convolutional_layer(batch,h,w,c,n,dilation,groups,size,stride,padding,activation, batch_normalize, params.net->optim);

    assign_weight_transform_norm_w_convolutional_layer(&l, wts);
    assign_quantization_norm_w_convolutional_layer(&l, qs);

    l.flipped = option_find_int_quiet(options, "flipped", 0);

    return l;
}

layer parse_crnn(list *options, size_params params)
{
    int output_filters = option_find_int(options, "output_filters",1);
    int hidden_filters = option_find_int(options, "hidden_filters",1);
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);

    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_crnn_layer(params.batch, params.w, params.h, params.c, hidden_filters, output_filters, params.time_steps, activation, batch_normalize, params.net->optim);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);

    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, output, params.time_steps, activation, batch_normalize, params.net->optim);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->optim);
    l.tanh = option_find_int_quiet(options, "tanh", 0);

    return l;
}

layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output", 1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize, params.net->optim);

    return l;
}

layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "logistic");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);

    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_connected_layer(params.batch, params.inputs, output, activation, batch_normalize, params.net->optim);
    return l;
}

layer parse_softmax(list *options, size_params params)
{
    int groups = option_find_int_quiet(options, "groups",1);
    layer l = make_softmax_layer(params.batch, params.inputs, groups);
    l.temperature = option_find_float_quiet(options, "temperature", 1);
    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    l.w = params.w;
    l.h = params.h;
    l.c = params.c;
    l.spatial = option_find_float_quiet(options, "spatial", 0);
    l.noloss =  option_find_int_quiet(options, "noloss", 0);
    return l;
}

layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);

    int total;
    float* anchors = 0;
    option_find_float_series(options, "anchors", &total, &anchors);
    assert(total % 2 == 0 && total > 0);
    total = total/2;

    int num;
    int* mask = 0;
    option_find_int_series(options, "mask", &num, &mask);
    if (mask == 0) num = total;
    int max_boxes = option_find_int_quiet(options, "max", 90);

    int softmax = option_find_int_quiet(options, "softmax", 0);
    int background = option_find_int_quiet(options, "background", 0);

    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes, anchors, softmax, background);
    assert(l.outputs == params.inputs);
    l.rescore = option_find_int_quiet(options, "rescore",0);
    l.bias_match = option_find_int_quiet(options, "bias_match", 1);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);

    l.coord_scale = option_find_float_quiet(options, "coord_scale", 1);
    l.object_scale = option_find_float_quiet(options, "object_scale", 1);
    l.noobject_scale = option_find_float_quiet(options, "noobject_scale", 1);
    l.class_scale = option_find_float_quiet(options, "class_scale", 1);
    l.warmup = option_find_int_quiet(options, "warmup", 0);
    l.blind = option_find_int_quiet(options, "blind", 0);
    l.logistic_derivative = option_find_int_quiet(options, "logistic_derivative", 0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) {
        l.softmax_tree = read_tree(tree_file);
    }
    
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) {
        l.map = read_map(map_file);
        l.map_gpu = cuda_make_int_array(l.map, l.classes);
    }

    free(anchors);
    if (mask) free(mask);

    return l;
}

layer parse_diff(list *options, size_params params, network* net)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    int truth_layer = option_find_int(options, "truth_layer", 0);
    int learn_layer = option_find_int(options, "learn_layer", -1);
    
    truth_layer = (truth_layer < 0) ? params.index + truth_layer : truth_layer;
    learn_layer = (learn_layer < 0) ? params.index + learn_layer : learn_layer;

    int skip_list_length;
    int* skip_list = 0;
    option_find_int_series(options, "skip_index", &skip_list_length, &skip_list);

    layer l = make_diff_layer(net, params.batch, truth_layer, learn_layer, type, skip_list, skip_list_length);
    l.mask_layer_softmax = option_find_int(options, "mask_layer_softmax", -1);

    return l;
}

layer parse_iseg(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int ids = option_find_int(options, "ids", 32);
    layer l = make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(l.outputs == params.inputs);
    return l;
}

layer parse_region(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);

    int num;
    float* anchors = 0;
    option_find_float_series(options, "anchors", &num, &anchors);
    assert(num % 2 == 0 && num > 0);
    num = num/2;

    int max_boxes = option_find_int_quiet(options, "max", 90);

    int softmax = option_find_int_quiet(options, "softmax", 0);
    int background = option_find_int_quiet(options, "background", 0);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, max_boxes, anchors, softmax, background);
    assert(l.outputs == params.inputs);

    l.rescore = option_find_int_quiet(options, "rescore",0);
    l.bias_match = option_find_int_quiet(options, "bias_match", 1);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);

    l.warmup = option_find_int_quiet(options, "warmup", 0);
    l.blind = option_find_int_quiet(options, "blind", 0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    free(anchors);

    return l;
}

layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    layer l = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    l.softmax = option_find_int(options, "softmax", 0);
    l.sqrt = option_find_int(options, "sqrt", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.forced = option_find_int(options, "forced", 0);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.reorg = option_find_int_quiet(options, "reorg", 0);

    l.blind = option_find_int_quiet(options, "blind", 0);

    return l;
}

layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    layer l = make_cost_layer(params.batch, params.inputs, type, scale);
    l.ratio =  option_find_float_quiet(options, "ratio",0);
    l.noobject_scale =  option_find_float_quiet(options, "noobj", 1);
    l.thresh =  option_find_float_quiet(options, "thresh",0);
    return l;
}

layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float_quiet(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);
    int flatten = option_find_int_quiet(options, "flatten",0);
    int extra = option_find_int_quiet(options, "extra",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.");

    layer l = make_reorg_layer(batch,w,h,c,stride,reverse, flatten, extra);
    return l;
}

layer parse_maxpool(list *options, size_params params)
{
    int spatial = option_find_int_quiet(options, "spatial", 0);
    int stride = 0;
    int size = 0;
    int padding = 0;
    int dilation = 0;

    if (!spatial) {
        stride = option_find_int(options, "stride", 1);
        size = option_find_int(options, "size", stride);
        padding = option_find_int_quiet(options, "padding", size-1);
        dilation = option_find_int_quiet(options, "dilation", 1);
    }

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    layer l = make_maxpool_layer(batch,h,w,c,size,stride,padding,dilation,spatial);
    return l;
}

layer parse_avgpool(list *options, size_params params)
{
    int spatial = option_find_int_quiet(options, "spatial", 0);
    int stride = 0;
    int size = 0;
    int padding = 0;
    int dilation = 0;

    if (!spatial) {
        stride = option_find_int(options, "stride", 1);
        size = option_find_int(options, "size", stride);
        padding = option_find_int_quiet(options, "padding", size-1);
        dilation = option_find_int_quiet(options, "dilation", 1);
    }

    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.");

    layer l = make_avgpool_layer(batch,w,h,c,size,stride,padding,dilation,spatial);
    return l;
}

layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    layer l = make_dropout_layer(params.batch, params.inputs, probability);
    l.out_w = params.w;
    l.out_h = params.h;
    l.out_c = params.c;
    return l;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c);
    return l;
}

layer parse_shortcut(list *options, size_params params, network *net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_shortcut_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);
    
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "linear");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);

    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    return s;
}


layer parse_l2norm(list *options, size_params params)
{
    layer l = make_l2norm_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}


layer parse_logistic(list *options, size_params params)
{
    layer l = make_logistic_layer(params.batch, params.inputs);
    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;
    return l;
}

layer parse_activation(list *options, size_params params)
{
    activation_scheme activation = {0};
    char *activation_s = option_find_str(options, "activation", "linear");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.h = l.out_h = params.h;
    l.w = l.out_w = params.w;
    l.c = l.out_c = params.c;

    return l;
}

layer parse_quantization(list *options, size_params params) {
    char *quantization_s = option_find_str(options, "scheme", "qs_none");
    quantization_scheme qs = {0};
    qs.type = get_quantization_scheme(quantization_s);
    qs.num_level = option_find_int(options, "num_level", 64);
    qs.step_size = option_find_float(options, "step_size", 0.125);
    qs.root = option_find_float_quiet(options, "root", 0.0);
    qs.zero_center = option_find_int_quiet(options, "zero_center", 0);

    layer l = make_quantization_layer(params.batch, params.w, params.h, params.c, qs);

    return l;
}

layer parse_upsample(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

layer parse_stack(list *options, size_params params, network *net)
{

    int stride = option_find_int(options, "stride",2);
    layer l = make_stack_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

layer parse_sr_flat(list *options, size_params params, network *net)
{
    float scale = option_find_int_quiet(options, "scale", 0);
    layer l = make_sr_flat_layer(params.batch, params.w, params.h, params.c, scale);
    return l;
}

layer parse_route(list *options, size_params params, network *net)
{
    char *ll = option_find(options, "layers");
    int len = strlen(ll);
    if(!ll) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (ll[i] == ',') ++n;
    }

    int *layers = calloc(n, sizeof(int));
    int *sizes = calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(ll);
        ll = strchr(ll, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
    }
    int batch = params.batch;

    layer l = make_route_layer(batch, n, layers, sizes);

    layer first = net->layers[layers[0]];
    l.out_w = first.out_w;
    l.out_h = first.out_h;
    l.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l.out_c += next.out_c;
        }else{
            l.out_h = l.out_w = l.out_c = 0;
        }
    }

    return l;
}

layer parse_scale_channels(list *options, size_params params, network* net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;
    int scale_wh = option_find_int_quiet(options, "scale_wh", 0);

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_scale_channels_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c, scale_wh);

    activation_scheme activation = {0};
    char *activation_s = option_find_str_quiet(options, "activation", "linear");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);

    s.activation = activation;
    return s;
}

//nghiant: sam layer, derived from darknet alexeyab
layer parse_sam(list *options, size_params params, network* net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net->layers[index];

    layer s = make_sam_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    activation_scheme activation = {0};
    char *activation_s = option_find_str_quiet(options, "activation", "linear");
    activation.type = get_activation(activation_s);
    activation.alpha = option_find_float_quiet(options, "act_alpha", 1);
    activation.beta = option_find_float_quiet(options, "act_beta", 1);
    
    s.activation = activation;
    return s;
}
//nghiant_end


LEARNING_RATE_POLICY get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    if (strcmp(s, "cyclical")==0) return CYCLICAL;
    if (strcmp(s, "pulse")==0) return PULSE;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

OPTIMIZER get_optimizer(char* s) {
    if (strcmp(s, "adam")==0) return ADAM;
    if (strcmp(s, "sgd")==0) return SGD;
    fprintf(stderr, "Couldn't find optimizer %s, going with SGD\n", s);
    return SGD;
}

char* get_optimizer_string(OPTIMIZER optim) {
    switch (optim) {
        case ADAM:
            return "adam";
        case SGD:
            return "sgd";
        default:
            break;
    }
    return "sgd";
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);

    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->notruth = option_find_int_quiet(options, "notruth",0);
    net->batch /= subdivs;
    net->batch *= net->time_steps;
    net->subdivisions = subdivs;
    net->random = option_find_int_quiet(options, "random", 0);

    char* optim_string = option_find_str_quiet(options, "optimizer", "sgd");
    net->optim = get_optimizer(optim_string);
    
    if (net->optim == SGD) {
        net->momentum = option_find_float(options, "momentum", .9);
    } else if (net->optim == ADAM) {
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .0000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->max_ratio = option_find_float_quiet(options, "max_ratio", (float) net->max_crop / net->w);
    net->min_ratio = option_find_float_quiet(options, "min_ratio", (float) net->min_crop / net->w);
    net->center = option_find_int_quiet(options, "center",0);
    net->clip = option_find_float_quiet(options, "clip", 0);

    //nghiant: data augmentation
    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);
    net->zoom = calloc(2, sizeof(float));
    net->zoom[0] = 1;
    net->zoom[1] = 1;
    option_find_float_series(options, "zoom", 0, &(net->zoom));
    net->hflip = option_find_float_quiet(options, "hflip", 0);
    net->vflip = option_find_float_quiet(options, "vflip", 0);
    net->solarize = option_find_float_quiet(options, "solarize", 0);
    net->posterize = option_find_float_quiet(options, "posterize", 0);
    net->noise = option_find_float_quiet(options, "noise", 0);
    net->jitter = option_find_float_quiet(options, "jitter", 0);
    net->cutout.prob = option_find_float_quiet(options, "cutout", 0);
    net->cutout.max_w = option_find_int_quiet(options, "cutout_max_w", 1);
    net->cutout.max_h = option_find_int_quiet(options, "cutout_max_h", 1);
    //nghiant_end

    char* init_s = option_find_str_quiet(options, "init", "he");
    net->initializer.type = get_initializer_type(init_s);
    net->initializer.mu = option_find_float_quiet(options, "init_mu", 0.0);
    net->initializer.sigma = option_find_float_quiet(options, "init_sigma", 0.0);
    net->initializer.alpha = option_find_float_quiet(options, "init_alpha", 0.0);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
    net->power = option_find_float_quiet(options, "power", 4);
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        if(!l || !p) error("STEPS policy must have steps and scales in cfg file");

        int len = strlen(l);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (l[i] == ',') ++n;
        }
        int *steps = calloc(n, sizeof(int));
        float *scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            int step    = atoi(l);
            float scale = atof(p);
            l = strchr(l, ',')+1;
            p = strchr(p, ',')+1;
            steps[i] = step;
            scales[i] = scale;
        }
        net->scales = scales;
        net->steps = steps;
        net->num_steps = n;
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
    } else if (net->policy == CYCLICAL) {
        net->step = option_find_int(options, "step", 1000);
        net->lower_bound = option_find_float(options, "lower_bound", 0.00001);
        net->upper_bound = option_find_float(options, "upper_bound", 0.001);
    } else if (net->policy == PULSE) {
        net->step = option_find_int(options, "step", 100);
        net->scale = option_find_float(options, "scale", 0.1);
    }
    net->max_batches = option_find_int(options, "max_batches", 0);
}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

network *parse_network_cfg(char *filename, int batch)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network *net = make_network(sections->size - 1);
    net->pre_transformable = 0;
    net->gpu_index = gpu_index;
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, net);
    if (batch > 0) net->batch = batch;

    params.h = net->h;
    params.w = net->w;
    params.c = net->c;
    params.inputs = net->inputs;
    params.batch = net->batch;
    params.time_steps = net->time_steps;
    params.net = net;

    size_t workspace_size = 0;
    n = n->next;
    int count = 0;
    free_section(s);
    fprintf(stderr, "layer     filters    size              input                output     BFLOP |   ACTIVATION SCHEME  |     QUANTIZATION SCHEME    |  WEIGHT TRANSFORM SCHEME  \n");

    char default_name[64];
        
    while(n){
        params.index = count;
        s = (section *)n->val;
        options = s->options;
        layer l = {0};
        int root_connect = option_find_int_quiet(options, "root_connect", 0);
        if (root_connect) {
            params.h = net->h;
            params.w = net->w;
            params.c = net->c;
            params.inputs = net->inputs;
            params.batch = net->batch;
        }
        if (count == 0 || root_connect) fprintf(stderr, "\033[0;32m>\033[0m%4d ", count);
        else fprintf(stderr, "%5d ", count);

        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
            net->total_CONV_w_size += l.nweights * 32.; //nghiant_20201123: support for rl training
        }else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        }else if(lt == LOCAL){
            l = parse_local(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == LOGXENT){
            l = parse_logistic(options, params);
        }else if(lt == L2NORM){
            l = parse_l2norm(options, params);
        }else if(lt == RNN){
            l = parse_rnn(options, params);
        }else if(lt == GRU){
            l = parse_gru(options, params);
        }else if (lt == LSTM) {
            l = parse_lstm(options, params);
        }else if(lt == CRNN){
            l = parse_crnn(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
        }else if(lt == REGION){
            l = parse_region(options, params);
        }else if(lt == YOLO){
            l = parse_yolo(options, params);
        }else if(lt == ISEG){
            l = parse_iseg(options, params);
        }else if(lt == DETECTION){
            l = parse_detection(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net->hierarchy = l.softmax_tree;
        }else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if(lt == REORG){
            l = parse_reorg(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == ROUTE){
            l = parse_route(options, params, net);
        }else if(lt == UPSAMPLE){
            l = parse_upsample(options, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
        }else if(lt == QUANTIZATION){
            l = parse_quantization(options, params);
        }else if(lt == DIFF){
            l = parse_diff(options, params, net);
        }else if(lt == STACK){
            l = parse_stack(options, params, net);
        }else if(lt == SR_FLAT){
            l = parse_sr_flat(options, params, net);
        }else if(lt == NORM_W_CONVOLUTIONAL){
            l = parse_norm_w_convolutional(options, params);
        }else if(lt == SAM){
            l = parse_sam(options, params, net);
            // net.layers[count-1].use_bin_output = 0;
            // net.layers[l.index].use_bin_output = 0;
            // net.layers[l.index].keep_delta_gpu = 1;
        }else if(lt == SCALE_CHANNELS){
            l = parse_scale_channels(options, params, net);
        }else if(lt == CUSTOM_CONV){
            l = parse_custom_conv(options, params);
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        sprintf(default_name, "layer_%d", params.index);
        l.self_index = params.index;
        sprintf(l.name, "%s", option_find_str_quiet(options, "name", default_name));
        l.frozen = option_find_int_quiet(options, "frozen", 0);
        l.root_connect = root_connect;
        print_layer(l);
        l.impact = option_find_float_quiet(options, "impact", 1.0);

        //initialization parsing
        char* init_s = option_find_str_quiet(options, "init", get_initializer_string(net->initializer.type));
        l.initializer.type = get_initializer_type(init_s);
        l.initializer.mu = option_find_float_quiet(options, "init_mu", net->initializer.mu);
        l.initializer.sigma = option_find_float_quiet(options, "init_sigma", net->initializer.sigma);
        l.initializer.alpha = option_find_float_quiet(options, "init_alpha", net->initializer.alpha);
        //initialization_end

        l.clip = net->clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net->layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
        if (l.weight_transform.type) net->pre_transformable = 1;
        if (l.cost) ++(net->n_loss);
    }

    net->sub_loss = calloc(net->n_loss, sizeof(float));
    net->sub_loss_id = calloc(net->n_loss, sizeof(int));

    print_optimizer(net);
    fprintf(stderr, "Loss Term: \033[0;33m%3d\033[0m\n", net->n_loss);
    int i, j;
    j = 0;
    for (i = 0; i < net->n; ++i) {
        if (net->layers[i].cost) {
            net->sub_loss_id[j++] = i;
            fprintf(stderr, " \033[0;33m%4.2f x %3d-%s\033[0m\n", net->layers[i].impact, i, get_layer_string(net->layers[i].type));
        } else if (net->layers[i].impact != 1.0) {
            fprintf(stderr, " %4.2f x %3d-%s\n", net->layers[i].impact, i, get_layer_string(net->layers[i].type));

        }
    }
    
    free_list(sections);
    layer out = get_network_output_layer(net);
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->delta = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));
    net->output_gpu = out.output_gpu;
    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
    net->delta_gpu = cuda_make_array(net->delta, net->inputs*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);

    if(workspace_size){
        net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
    }
    fprintf(stderr, "Total conv size: %f\n", net->total_CONV_w_size);
    return net;
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

void save_convolutional_weights(layer l, FILE *fp)
{
    pull_convolutional_layer(l);

    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_norm_w_convolutional_weights(layer l, FILE *fp)
{
    pull_norm_w_convolutional_layer(l);

    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights_v, sizeof(float), l.nweights_v, fp);
    fwrite(l.weights_g, sizeof(float), l.nweights_g, fp);
}

void save_batchnorm_weights(layer l, FILE *fp)
{
    pull_batchnorm_layer(l);

    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
    pull_connected_layer(l);

    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network *net, char *filename, int cutoff)
{
    cuda_set_device(net->gpu_index);

    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = 0;
    int minor = 2;
    int revision = 0;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(net->seen, sizeof(size_t), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontsave) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            save_convolutional_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if (l.type == LSTM) {
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
        } if (l.type == GRU) {
            save_connected_weights(*(l.wz), fp);
            save_connected_weights(*(l.wr), fp);
            save_connected_weights(*(l.wh), fp);
            save_connected_weights(*(l.uz), fp);
            save_connected_weights(*(l.ur), fp);
            save_connected_weights(*(l.uh), fp);
        } if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
            pull_local_layer(l);

            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        } if(l.type == NORM_W_CONVOLUTIONAL){
            save_norm_w_convolutional_weights(l, fp);
        }
    }
    fclose(fp);
}
void save_weights(network *net, char *filename)
{
    save_weights_upto(net, filename, net->n);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float *transpose = calloc(rows*cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
    }

    push_connected_layer(l);
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);

    push_batchnorm_layer(l);
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.numload) l.n = l.numload;
    int num = l.c/l.groups*l.n*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.weights, sizeof(float), num, fp);
    if (l.flipped) {
        transpose_matrix(l.weights, l.c*l.size*l.size, l.n);
    }

    push_convolutional_layer(l);
}

void load_norm_w_convolutional_weights(layer l, FILE *fp)
{
    if(l.numload) l.n = l.numload;
    int num_v = l.nweights_v;
    int num_g = l.nweights_g;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.weights_v, sizeof(float), num_v, fp);
    fread(l.weights_g, sizeof(float), num_g, fp);

    push_norm_w_convolutional_layer(l);
    norm_w_normalize_gpu(l.weights_v_gpu, l.weights_v_norm_gpu, l.nweights_v/l.n, l.n);
    norm_w_reform_weights_gpu(l.weights_gpu, l.weights_v_gpu, l.weights_v_norm_gpu, l.weights_g_gpu, l.nweights_v/l.n, l.n);
}

void load_modular_weights(network* net, char* modules) {
    list* module_list = read_module_cfg(modules);

    int i;
    char sid[64];
    int header = 1;
    for (i = 0; i < net->n; ++i) {

        layer l = net->layers[i];
        sprintf(sid, "%s", l.name);
        char* lmodule = option_find_str_quiet(module_list, sid, 0);
        if (!lmodule) continue;

        if (l.dontload) continue;
        if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_convolutional_weights(l, mfp);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if(l.type == CONNECTED){
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_connected_weights(l, mfp, 1);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if(l.type == BATCHNORM){
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_batchnorm_weights(l, mfp);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if(l.type == CRNN){
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_convolutional_weights(*(l.input_layer), mfp);
                load_convolutional_weights(*(l.self_layer), mfp);
                load_convolutional_weights(*(l.output_layer), mfp);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if(l.type == RNN){
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_connected_weights(*(l.input_layer), mfp, 0);
                load_connected_weights(*(l.self_layer), mfp, 0);
                load_connected_weights(*(l.output_layer), mfp, 0);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if (l.type == LSTM) {
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_connected_weights(*(l.wi), mfp, 0);
                load_connected_weights(*(l.wf), mfp, 0);
                load_connected_weights(*(l.wo), mfp, 0);
                load_connected_weights(*(l.wg), mfp, 0);
                load_connected_weights(*(l.ui), mfp, 0);
                load_connected_weights(*(l.uf), mfp, 0);
                load_connected_weights(*(l.uo), mfp, 0);
                load_connected_weights(*(l.ug), mfp, 0);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if (l.type == GRU) {
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_connected_weights(*(l.wz), mfp, 0);
                load_connected_weights(*(l.wr), mfp, 0);
                load_connected_weights(*(l.wh), mfp, 0);
                load_connected_weights(*(l.uz), mfp, 0);
                load_connected_weights(*(l.ur), mfp, 0);
                load_connected_weights(*(l.uh), mfp, 0);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if(l.type == LOCAL){
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                int locations = l.out_w*l.out_h;
                int size = l.size*l.size*l.c*l.n*locations;
                fread(l.biases, sizeof(float), l.outputs, mfp);
                fread(l.weights, sizeof(float), size, mfp);

                push_local_layer(l);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
        if (l.type == NORM_W_CONVOLUTIONAL) {
            if (header) fprintf(stderr, "Modules:\n"), header = 0;
            FILE* mfp = fopen(lmodule, "rb");
            fprintf(stderr, " layer %3d ... ", i);
            if (mfp != NULL) {
                load_norm_w_convolutional_weights(l, mfp);
                fclose(mfp);
                fprintf(stderr, "\033[0;32m%s\033[0m\n", lmodule);
            } else {
                fprintf(stderr, "\033[0;31m%s\033[0m\n", lmodule);
            }
        }
    }
    option_unused(module_list);
}

void load_weights_upto(network *net, char *filename, int start, int cutoff)
{
    cuda_set_device(net->gpu_index);

    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        file_error(filename);
    }
    fprintf(stderr, "Weights: \033[0;32m%s\033[0m\n", filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
        fread(net->seen, sizeof(size_t), 1, fp);
    } else {
        int iseen = 0;
        fread(&iseen, sizeof(int), 1, fp);
        *net->seen = iseen;
    }
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = start; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];

        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            load_convolutional_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, 0);
            load_connected_weights(*(l.self_layer), fp, 0);
            load_connected_weights(*(l.output_layer), fp, 0);
        }
        if (l.type == LSTM) {
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
        }
        if (l.type == GRU) {
            load_connected_weights(*(l.wz), fp, transpose);
            load_connected_weights(*(l.wr), fp, transpose);
            load_connected_weights(*(l.wh), fp, transpose);
            load_connected_weights(*(l.uz), fp, transpose);
            load_connected_weights(*(l.ur), fp, transpose);
            load_connected_weights(*(l.uh), fp, transpose);
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);

            push_local_layer(l);
        }
        if(l.type == NORM_W_CONVOLUTIONAL){
            load_norm_w_convolutional_weights(l, fp);
        }
    }
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, 0, net->n);
}

char* get_weight_transform_scheme_string(WEIGHT_TRANSFORM_SCHEME type) {
    switch(type){
        case WTS_MAX_SHIFTER:
            return "MAX_SHIFTER";
        case WTS_MEAN_SHIFTER:
            return "MEAN_SHIFTER";
        case WTS_UNIFORM:
            return "UNIFORM";
        case WTS_SHIFTER:
            return "SHIFTER";
        case WTS_SCALE_LINEAR:
            return "SCALE_LINEAR";
        case WTS_NONE:
            return "NONE";
        default:
            return "NONE";
    }
}

void print_weight_transform_scheme_summary(weight_transform_scheme wts) {
    char buff[64];
    int arg_1 = 5;
    int arg_2 = 5;
    int arg_2d= 3;
    switch(wts.type){
        case WTS_MAX_SHIFTER:
        case WTS_MEAN_SHIFTER:
        case WTS_SHIFTER:
        case WTS_SCALE_LINEAR:
            sprintf(buff, "%s %*d %*s-", get_weight_transform_scheme_string(wts.type), arg_1, wts.num_level, arg_2-1, "");
            break;
        case WTS_UNIFORM:
            sprintf(buff, "%s %*d %*.*f", get_weight_transform_scheme_string(wts.type), arg_1, wts.num_level, arg_2, arg_2d, wts.step_size);
            break;
        case WTS_NONE:
        default:
            sprintf(buff, "%s %*s- %*s-", get_weight_transform_scheme_string(wts.type), arg_1-1, "", arg_2-1, "");
            break;
    }
    fprintf(stderr, " %24s ", buff);
    return;
}

WEIGHT_TRANSFORM_SCHEME get_weight_transform_scheme(char *s) {
    if (strcmp(s, "wts_max_shifter")==0) return WTS_MAX_SHIFTER;
    if (strcmp(s, "wts_mean_shifter")==0) return WTS_MEAN_SHIFTER;
    if (strcmp(s, "wts_uniform")==0) return WTS_UNIFORM;
    if (strcmp(s, "wts_shifter")==0) return WTS_SHIFTER;
    if (strcmp(s, "wts_scale_linear")==0) return WTS_SCALE_LINEAR;
    if (strcmp(s, "wts_none")==0) return WTS_NONE;
    fprintf(stderr, "Couldn't find weight transformation scheme %s, going with WTS_NONE\n", s);
    return WTS_NONE;
}

void print_layer(layer l) {
    LAYER_TYPE lt = l.type;
        if(lt == CONVOLUTIONAL){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            if (l.groups > 1 && l.dilation > 1) {
                fprintf(stderr, "dgconv\033[0m%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            } else if (l.groups > 1) {
                fprintf(stderr, "gconv\033[0m %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            } else if (l.dilation > 1) {
                fprintf(stderr, "dconv\033[0m %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            } else {
                fprintf(stderr, "conv\033[0m  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            }
            fprintf(stderr, "|");
            print_activation_summary(l.activation);
            fprintf(stderr, "|");
            print_quantization_scheme_summary(l.quantization);
            fprintf(stderr, "|");
            print_weight_transform_scheme_summary(l.weight_transform);
            fprintf(stderr, "\n");
        }else if(lt == NORM_W_CONVOLUTIONAL){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            if (l.groups > 1 && l.dilation > 1) {
                fprintf(stderr, "dgnonv\033[0m%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            } else if (l.groups > 1) {
                fprintf(stderr, "gnonv\033[0m %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            } else if (l.dilation > 1) {
                fprintf(stderr, "dnonv\033[0m %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            } else {
                fprintf(stderr, "nonv\033[0m  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f ", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000.);
            }
            fprintf(stderr, "|");
            print_activation_summary(l.activation);
            fprintf(stderr, "|");
            print_quantization_scheme_summary(l.quantization);
            fprintf(stderr, "|");
            print_weight_transform_scheme_summary(l.weight_transform);
            fprintf(stderr, "\n");
        }else if(lt == DECONVOLUTIONAL){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "deconv\033[0m%5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.n, l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
        }else if(lt == LOCAL){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "local\033[0m %d x %d x %d image, %d filters -> %d x %d x %d image\n", l.h,l.w,l.c,l.n, l.out_h, l.out_w, l.n);
        }else if(lt == ACTIVE){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "active\033[0m %d inputs\n", l.inputs);
        }else if(lt == LOGXENT){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "logistic x entropy\033[0m                             %4d\n",  l.inputs);
        }else if(lt == L2NORM){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "l2norm\033[0m                                         %4d\n",  l.inputs);
        }else if(lt == RNN){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "rnn\033[0m  %d inputs, %d outputs\n", l.inputs, l.outputs);
            fprintf(stderr, "\t\t");
            print_layer(*(l.input_layer));
            fprintf(stderr, "\t\t");
            print_layer(*(l.self_layer));
            fprintf(stderr, "\t\t");
            print_layer(*(l.output_layer));
        }else if(lt == GRU){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "gru\033[0m  %d inputs, %d outputs\n", l.inputs, l.outputs);
            fprintf(stderr, "\t\t");
            print_layer(*(l.uz));
            fprintf(stderr, "\t\t");
            print_layer(*(l.wz));
            fprintf(stderr, "\t\t");
            print_layer(*(l.ur));
            fprintf(stderr, "\t\t");
            print_layer(*(l.wr));
            fprintf(stderr, "\t\t");
            print_layer(*(l.uh));
            fprintf(stderr, "\t\t");
            print_layer(*(l.wh));
        }else if (lt == LSTM) {
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "lstm\033[0m %d inputs, %d outputs\n", l.inputs, l.outputs);
            fprintf(stderr, "\t\t");
            print_layer(*(l.uf));
            fprintf(stderr, "\t\t");
            print_layer(*(l.ui));
            fprintf(stderr, "\t\t");
            print_layer(*(l.ug));
            fprintf(stderr, "\t\t");
            print_layer(*(l.uo));
            fprintf(stderr, "\t\t");
            print_layer(*(l.wf));
            fprintf(stderr, "\t\t");
            print_layer(*(l.wi));
            fprintf(stderr, "\t\t");
            print_layer(*(l.wg));
            fprintf(stderr, "\t\t");
            print_layer(*(l.wo));
        }else if(lt == CRNN){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "crnn\033[0m %d x %d x %d image, %d filters\n", l.h,l.w,l.c,l.out_c);
            fprintf(stderr, "\t\t");
            print_layer(*(l.input_layer));
            fprintf(stderr, "\t\t");
            print_layer(*(l.self_layer));
            fprintf(stderr, "\t\t");
            print_layer(*(l.output_layer));
        }else if(lt == CONNECTED){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "connected\033[0m                            %4d  ->  %4d\n", l.inputs,l.outputs);
        }else if(lt == CROP){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "crop\033[0m %d x %d -> %d x %d x %d image\n", l.h,l.w,l.out_h,l.out_w,l.c);
        }else if(lt == COST){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "cost\033[0m                                           %4d\n",  l.inputs);
        }else if(lt == REGION){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "region\033[0m   %2d  %s  %s", l.n, l.softmax ? "\033[0;33msoftmax\033[0m" : "\033[0;33mlogistic\033[0m", l.background ? "\033[0;32mbackground\033[0m" : "\033[0;31mbackground\033[0m");
            if (l.warmup > 0) fprintf(stderr, "  \033[0;32mwarmup: %d\033[0m", l.warmup);
            if (l.blind > 0) fprintf(stderr, "  \033[0;32mblind\033[0m");
            fprintf(stderr, "\n");
        }else if(lt == YOLO){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "yolo\033[0m  %2d/%2d  %s  %s", l.n, l.total, l.softmax ? "\033[0;33msoftmax\033[0m" : "\033[0;33mlogistic\033[0m", l.background ? "\033[0;32mbackground\033[0m" : "\033[0;31mbackground\033[0m");
            if (l.warmup > 0) fprintf(stderr, "  \033[0;32mwarmup: %d\033[0m", l.warmup);
            if (l.blind > 0) fprintf(stderr, "  \033[0;32mblind\033[0m");
            if (l.logistic_derivative > 0) fprintf(stderr, "  \033[0;32mlogistic_derivative\033[0m");
            fprintf(stderr, "\n");
        }else if(lt == ISEG){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "iseg\033[0m\n");
        }else if(lt == DETECTION){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "detection\033[0m\n");
            if (l.blind > 0) fprintf(stderr, "  \033[0;32mblind\033[0m");
        }else if(lt == SOFTMAX){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "softmax\033[0m                                        %4d\n",  l.inputs);
        }else if(lt == NORMALIZATION){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "normalization\033[0m %d x %d x %d image, %d size\n", l.w,l.h,l.c,l.size);
        }else if(lt == BATCHNORM){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "batch norm\033[0m %d x %d x %d image\n", l.w,l.h,l.c);
        }else if(lt == MAXPOOL){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            if (!l.spatial) fprintf(stderr, "max\033[0m          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
            else fprintf(stderr, "max\033[0m       \033[0;33mspatial\033[0m       %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
        }else if(lt == REORG){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            if(l.extra){
                fprintf(stderr, "reorg\033[0m              %4d   ->  %4d\n",  l.inputs, l.outputs);
            } else {
                fprintf(stderr, "reorg\033[0m              /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
            }
        }else if(lt == AVGPOOL){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            if (!l.spatial) fprintf(stderr, "avg\033[0m          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.size, l.size, l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
            else fprintf(stderr, "avg\033[0m       \033[0;33mspatial\033[0m       %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
        }else if(lt == ROUTE){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr,"route\033[0m ");
            int i;
            for(i = 0; i < l.n; ++i){
                fprintf(stderr," %d", l.input_layers[i]);
            }
            fprintf(stderr, "\n");
        }else if(lt == UPSAMPLE){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            if(l.reverse) fprintf(stderr, "downsample\033[0m         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
            else fprintf(stderr, "upsample\033[0m           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
        }else if(lt == SHORTCUT){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "res\033[0m  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d        ",l.index, l.w,l.h,l.c, l.out_w,l.out_h,l.out_c);
            fprintf(stderr, "|");
            print_activation_summary(l.activation);
            fprintf(stderr, "\n");
        }else if(lt == DROPOUT){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "dropout\033[0m       p = %.2f               %4d  ->  %4d\n", l.probability, l.inputs, l.inputs);
        }else if(lt == QUANTIZATION){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "quantization\033[0m ");
            print_quantization_scheme_summary(l.quantization);
            fprintf(stderr, "\n");
        }else if(lt == DIFF){
            if (l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "diff\033[0m  %5d -> %5d    %s", l.input_layers[0], l.input_layers[1], get_cost_string(l.cost_type));
            if (l.n > 0) fprintf(stderr, "  \033[0;33mskip %d channel(s)\033[0m", l.n);
            fprintf(stderr, "\n");
        }else if(lt == STACK){
            if(l.frozen) fprintf(stderr, "\033[0;36m");
            if(l.reverse) fprintf(stderr, "compress\033[0m         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
            else fprintf(stderr, "stack\033[0m              %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.stride, l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
        }else if(lt == SR_FLAT){
            if(l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "sr_flat\033[0m                 %4d x%4d x%4d   ->  %4d x%4d x%4d\n", l.w, l.h, l.c, l.out_w, l.out_h, l.out_c);
        }else if(lt == SAM){
            if(l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "sam\033[0m\n");
        }else if(lt == SCALE_CHANNELS){
            if(l.frozen) fprintf(stderr, "\033[0;36m");
            fprintf(stderr, "scale_channels\033[0m\n");
        }else{
            fprintf(stderr, "\033[0;31mimplement in print_layer() in parser.c\033[0m");
        }
}

void print_optimizer(network* net) {
    switch (net->optim) {
        case ADAM:
            fprintf(stderr, "Optimizer: \033[0;32mADAM\033[0m   \033[0;33mB1: %.3f\033[0m   \033[0;33mB2: %.3f\033[0m   \033[0;33meps: %.7f\033[0m\n", net->B1, net->B2, net->eps);
            return;
        case SGD:
            fprintf(stderr, "Optimizer: \033[0;32mSGD\033[0m    \033[0;33mmomentum: %.3f\033[0m\n", net->momentum);
            return;
        default:
            fprintf(stderr, "Unknown optimizer\n");
            return;
    }
}