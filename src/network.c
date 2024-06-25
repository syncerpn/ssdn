#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "lstm_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "logistic_layer.h"
#include "parser.h"
#include "data.h"
#include "quantization_layer.h"
#include "diff_layer.h"
#include "initializer.h"
#include "sr_flat_layer.h"
#include "stack_layer.h"
#include "norm_w_convolutional_layer.h"
#include "sam_layer.h"
#include "scale_channels_layer.h"

load_args get_base_args(network *net)
{
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.size = net->w;

    args.min = net->min_crop;
    args.max = net->max_crop;
    //data augmentation
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.center = net->center;
    args.saturation = net->saturation;
    args.hue = net->hue;
    
    args.zoom = calloc(2, sizeof(float));
    args.zoom[0] = net->zoom[0];
    args.zoom[1] = net->zoom[1];

    args.hflip = net->hflip;
    args.vflip = net->vflip;
    args.solarize = net->solarize;
    args.posterize = net->posterize;
    args.noise = net->noise;
    args.jitter = net->jitter;
    args.cutout = net->cutout;
    return args;
}

void initialize_network(network* net) {
    int i;
    fprintf(stderr, "Global Initialization: \033[0;32m%s\033[0m\n", get_initializer_string(net->initializer.type));
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        
        if(l.type == CONVOLUTIONAL){
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_convolutional_layer(l, l.initializer);
        }
        if(l.type == DECONVOLUTIONAL){
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_deconvolutional_layer(l, l.initializer);
        }
        if(l.type == CONNECTED){
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_connected_layer(l, l.initializer);
        }
        if(l.type == CRNN){
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_crnn_layer(l, l.initializer);
        }
        if(l.type == RNN){
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_rnn_layer(l, l.initializer);
        }
        if (l.type == LSTM) {
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_lstm_layer(l, l.initializer);
        }
        if (l.type == GRU) {
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_gru_layer(l, l.initializer);
        }
        if(l.type == LOCAL){
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_local_layer(l, l.initializer);
        }
        if(l.type == NORM_W_CONVOLUTIONAL){
            if (l.initializer.type != net->initializer.type) fprintf(stderr, " layer %3d: \033[0;32m%s\033[0m\n", i, get_initializer_string(l.initializer.type));
            init_norm_w_convolutional_layer(l, l.initializer);
        }
    }
}

network *load_network(char *cfg, char *weights, char* modules, int clear, int batch)
{
    network *net = parse_network_cfg(cfg, batch);
    initialize_network(net);

    if(weights && weights[0] != 0){
        load_weights(net, weights);
    }

    if (modules) load_modular_weights(net, modules);

    if(clear) (*net->seen) = 0;
    return net;
}

size_t get_current_batch(network *net)
{
    size_t batch_num = (*net->seen)/(net->batch*net->subdivisions);
    return batch_num;
}

void reset_network_state(network *net, int b)
{
    int i;
    for (i = 0; i < net->n; ++i) {

        layer l = net->layers[i];
        if(l.state_gpu){
            fill_gpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
        }
        if(l.h_gpu){
            fill_gpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
        }
    }
}

void reset_rnn(network *net)
{
    reset_network_state(net, 0);
}

float get_current_rate(network *net)
{
    size_t batch_num = get_current_batch(net);
    int i;
    float rate;
    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);
    switch (net->policy) {
        case CONSTANT:
            return net->learning_rate;
        case STEP:
            return net->learning_rate * pow(net->scale, batch_num/net->step);
        case STEPS:
            rate = net->learning_rate;
            for(i = 0; i < net->num_steps; ++i){
                if(net->steps[i] > batch_num) return rate;
                rate *= net->scales[i];
            }
            return rate;
        case EXP:
            return net->learning_rate * pow(net->gamma, batch_num);
        case POLY:
            return net->learning_rate * pow(1 - (float)batch_num / net->max_batches, net->power);
        case RANDOM:
            return net->learning_rate * pow(rand_uniform(0,1), net->power);
        case SIG:
            return net->learning_rate * (1./(1.+exp(net->gamma*(batch_num - net->step))));
        //nghiant
        case CYCLICAL:
            if ((batch_num % (2 * net->step)) < net->step) return net->lower_bound + (net->upper_bound - net->lower_bound) * (float)(batch_num % net->step) / (float)(net->step);
            else return net->upper_bound - (net->upper_bound - net->lower_bound) * (float)(batch_num % net->step) / (float)(net->step);
        case PULSE:
            return net->learning_rate * (batch_num/net->step % 2 == 0 ? 1 : net->scale);
        //nghiant_end
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net->learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case LSTM:
	       return "lstm";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case YOLO:
            return "yolo";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        case UPSAMPLE:
            return "upsample";
            //nghiant
        case QUANTIZATION:
            return "quantization";
        case DIFF:
            return "diff";
        case NORM_W_CONVOLUTIONAL:
            return "norm_w_convolutional";
        case SAM:
            return "sam";
        case SCALE_CHANNELS:
            return "scale_channels";
            //nghiant_end
        default:
            break;
    }
    return "none";
}

network *make_network(int n)
{
    network *net = calloc(1, sizeof(network));
    net->n = n;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    net->n_loss = 0;
    net->pre_transform = 0;

    return net;
}

void calc_network_cost(network *netp)
{
    network net = *netp;
    int i, j;
    float sum = 0;
    for(j = 0; j < net.n_loss; ++j){
        i = net.sub_loss_id[j];
        sum += net.layers[i].cost[0];
        net.sub_loss[j] += net.layers[i].cost[0];
    }
    *net.cost = sum/net.n_loss;
}

float train_network_datum(network *net)
{
    *net->seen += net->batch;
    net->train = 1;
    forward_network(net);
    backward_network(net);
    float error = *net->cost;
    if(((*net->seen)/net->batch)%net->subdivisions == 0) update_network(net);
    return error;
}

float train_network(network *net, data d)
{
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;

    int i;
    float sum = 0;
    //nghiant: multi loss terms; reset first
    int il;
    for (il = 0; il < net->n_loss; ++il) {
    	net->sub_loss[il] = 0;
    }

    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, net->input, net->truth);
        float err = train_network_datum(net);
        sum += err;
    }

    //nghiant: multi loss terms; averaging
    for (il = 0; il < net->n_loss; ++il) {
    	net->sub_loss[il] /= (n*batch);
    }

    return (float)sum/(n*batch);
}

int resize_network(network *net, int w, int h)
{
    cuda_set_device(net->gpu_index);
    cuda_free(net->workspace);

    int i;

    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;

    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
        }else if(l.type == YOLO){
            resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
        }else if(l.type == SHORTCUT){
            resize_shortcut_layer(&l, w, h);
        }else if(l.type == UPSAMPLE){
            resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else if(l.type == DIFF){
            resize_diff_layer(&l, net);
        }else if(l.type == DROPOUT){
            resize_dropout_layer(&l, inputs);
        }else if(l.type == ACTIVE){
            resize_activation_layer(&l, inputs);
        }else if(l.type == LOGXENT){
            resize_logistic_layer(&l, inputs);
        }else if(l.type == L2NORM){
            resize_logistic_layer(&l, inputs);
        }else if(l.type == BATCHNORM){
            resize_batchnorm_layer(&l, w, h);
        }else if(l.type == SR_FLAT){
            resize_sr_flat_layer(&l, w, h);
        }else if(l.type == STACK){
            resize_stack_layer(&l, w, h);
        }else if(l.type == QUANTIZATION){
            resize_quantization_layer(&l, w, h);
        }else if(l.type == NORM_W_CONVOLUTIONAL){
            resize_norm_w_convolutional_layer(&l, w, h);
        }else if(l.type == SAM){
            resize_sam_layer(&l, w, h);
        }else{
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if(l.workspace_size > 2000000000) assert(0);
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
    layer out = get_network_output_layer(net);
    net->inputs = net->layers[0].inputs;
    net->outputs = out.outputs;
    net->truths = out.outputs;
    if(net->layers[net->n-1].truths) net->truths = net->layers[net->n-1].truths;
    net->output = out.output;
    free(net->input);
    free(net->delta);
    free(net->truth);
    net->input = calloc(net->inputs*net->batch, sizeof(float));
    net->delta = calloc(net->inputs*net->batch, sizeof(float));
    net->truth = calloc(net->truths*net->batch, sizeof(float));

    cuda_free(net->input_gpu);
    cuda_free(net->delta_gpu);
    cuda_free(net->truth_gpu);
    net->input_gpu = cuda_make_array(net->input, net->inputs*net->batch);
    net->delta_gpu = cuda_make_array(net->delta, net->inputs*net->batch);
    net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
    if(workspace_size){
        net->workspace = cuda_make_array(0, (workspace_size-1)/sizeof(float)+1);
    }

    return 0;
}

image get_network_image_layer(network *net, int i)
{
    layer l = net->layers[i];

    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network *net)
{
    int i;
    for(i = net->n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network *net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net->n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network *net, int k, int *index)
{
    top_k(net->output, net->outputs, k, index);
}


float *network_predict(network *net, float *input)
{
    network orig = *net;
    net->input = input;
    net->truth = 0;
    net->train = 0;
    net->delta = 0;
    forward_network(net);
    float *out = net->output;
    *net = orig;
    return out;
}

int num_detections(network *net, float thresh)
{
    int i;
    int s = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO && (!(l.blind))){
            s += yolo_num_detections(l, thresh);
        }
        if((l.type == DETECTION || l.type == REGION) && (!(l.blind))){
            s += l.w*l.h*l.n;
        }
    }
    return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
    layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if(num) *num = nboxes;
    detection *dets = calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i){
        dets[i].prob = calloc(l.classes, sizeof(float));
        if(l.coords > 4){
            dets[i].mask = calloc(l.coords-4, sizeof(float));
        }
    }
    return dets;
}

void copy_detection(detection* dst, detection* src, int n_classes, size_t nboxes) {
    memcpy(dst, src, nboxes * sizeof(detection));
    int i;
    for (i = 0; i < nboxes; ++i) {
        dst[i].prob = calloc(n_classes, sizeof(float));
        memcpy(dst[i].prob, src[i].prob, n_classes * sizeof(float));
    }
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j;
    for(j = 0; j < net->n; ++j){
        layer l = net->layers[j];
        if(l.type == YOLO && (!(l.blind))){
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += count;
        }
        if(l.type == REGION && (!(l.blind))){
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION && (!(l.blind))){
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
        if(dets[i].mask) free(dets[i].mask);
    }
    free(dets);
}

matrix network_predict_data(network *net, data test)
{
    int i,j,b;
    int k = net->outputs;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

float *network_accuracies(network *net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

layer get_network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type == COST || net->layers[i].type == DIFF) continue;
        else break;
    }
    return net->layers[i];
}

void free_network(network *net)
{
    int i;
    for(i = 0; i < net->n; ++i){
        free_layer(net->layers[i]);
    }
    free(net->layers);
    if(net->input) free(net->input);
    if(net->truth) free(net->truth);
    if(net->delta) free(net->delta);
    if(net->input_gpu) cuda_free(net->input_gpu);
    if(net->truth_gpu) cuda_free(net->truth_gpu);
    if(net->delta_gpu) cuda_free(net->delta_gpu);
    free(net);
}

layer network_output_layer(network *net)
{
    int i;
    for(i = net->n - 1; i >= 0; --i){
        if(net->layers[i].type != COST) break;
    }
    return net->layers[i];
}

void forward_network(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    cuda_push_array(net.input_gpu, net.input, net.inputs*net.batch);

    float* root_input = net.input;
    float* root_input_gpu = net.input_gpu;
    
    if(net.truth){
        cuda_push_array(net.truth_gpu, net.truth, net.truths*net.batch);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        net.index = i;
        layer l = net.layers[i];
        if (l.delta) {
            fill_gpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        
        //connect to input
        if (l.root_connect) {
        	net.input = root_input;
        	net.input_gpu = root_input_gpu;
        }

        l.forward_gpu(l, net);
        net.input_gpu = l.output_gpu;
        net.input = l.output;
        if(l.truth) {
            net.truth_gpu = l.output_gpu;
            net.truth = l.output;
        }
    }

    if (!net.train) pull_network_output(netp);
    calc_network_cost(netp);
}

void backward_network(network *netp)
{
    int i;
    
    network net = *netp;
    network orig = net;
    cuda_set_device(net.gpu_index);
    for(i = net.n-1; i >= 0; --i){
        layer l = net.layers[i];
        if(l.stopbackward) break;
        if(i == 0 || l.root_connect == 1){
            net = orig;
        }else{
            layer prev = net.layers[i-1];
            net.input = prev.output;
            net.delta = prev.delta;
            net.input_gpu = prev.output_gpu;
            net.delta_gpu = prev.delta_gpu;
        }
        net.index = i;
        l.backward_gpu(l, net);
    }
}

void update_network(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    update_args a = {0};
    a.optim = net.optim;
    a.batch = net.batch*net.subdivisions;
    a.learning_rate = get_current_rate(netp);
    a.momentum = net.momentum;
    a.decay = net.decay;
    a.B1 = net.B1;
    a.B2 = net.B2;
    a.eps = net.eps;
    ++*net.t;
    a.t = (*net.t);

    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update_gpu && !(l.frozen)){
            l.update_gpu(l, a);
        }
    }
}

void harmless_update_network_gpu(network *netp)
{
    network net = *netp;
    cuda_set_device(net.gpu_index);
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.weight_updates_gpu) fill_gpu(l.nweights, 0, l.weight_updates_gpu, 1);
        if(l.bias_updates_gpu) fill_gpu(l.nbiases, 0, l.bias_updates_gpu, 1);
        if(l.scale_updates_gpu) fill_gpu(l.nbiases, 0, l.scale_updates_gpu, 1);
    }
}

typedef struct {
    network *net;
    data d;
    float *err;
} train_args;

void *train_thread(void *ptr)
{
    train_args args = *(train_args*)ptr;
    free(ptr);
    cuda_set_device(args.net->gpu_index);
    *args.err = train_network(args.net, args.d);
    return 0;
}

pthread_t train_network_in_thread(network *net, data d, float *err)
{
    pthread_t thread;
    train_args *ptr = (train_args *)calloc(1, sizeof(train_args));
    ptr->net = net;
    ptr->d = d;
    ptr->err = err;
    if(pthread_create(&thread, 0, train_thread, ptr)) error("Thread creation failed");
    return thread;
}

void merge_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL) {
        axpy_cpu(l.n, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.nweights, 1, l.weight_updates, 1, base.weights, 1);
        if (l.scales) {
            axpy_cpu(l.n, 1, l.scale_updates, 1, base.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        axpy_cpu(l.outputs, 1, l.bias_updates, 1, base.biases, 1);
        axpy_cpu(l.outputs*l.inputs, 1, l.weight_updates, 1, base.weights, 1);
    }
}

void scale_weights(layer l, float s)
{
    if (l.type == CONVOLUTIONAL) {
        scal_cpu(l.n, s, l.biases, 1);
        scal_cpu(l.nweights, s, l.weights, 1);
        if (l.scales) {
            scal_cpu(l.n, s, l.scales, 1);
        }
    } else if(l.type == CONNECTED) {
        scal_cpu(l.outputs, s, l.biases, 1);
        scal_cpu(l.outputs*l.inputs, s, l.weights, 1);
    }
}


void pull_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.n);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.nweights);
        if(l.scales) cuda_pull_array(l.scales_gpu, l.scale_updates, l.n);
    } else if(l.type == CONNECTED){
        cuda_pull_array(l.biases_gpu, l.bias_updates, l.outputs);
        cuda_pull_array(l.weights_gpu, l.weight_updates, l.outputs*l.inputs);
    }
}

void push_weights(layer l)
{
    if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
        cuda_push_array(l.biases_gpu, l.biases, l.n);
        cuda_push_array(l.weights_gpu, l.weights, l.nweights);
        if(l.scales) cuda_push_array(l.scales_gpu, l.scales, l.n);
    } else if(l.type == CONNECTED){
        cuda_push_array(l.biases_gpu, l.biases, l.outputs);
        cuda_push_array(l.weights_gpu, l.weights, l.outputs*l.inputs);
    }
}

void distribute_weights(layer l, layer base)
{
    if (l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL) {
        cuda_push_array(l.biases_gpu, base.biases, l.n);
        cuda_push_array(l.weights_gpu, base.weights, l.nweights);
        if (base.scales) cuda_push_array(l.scales_gpu, base.scales, l.n);
    } else if (l.type == CONNECTED) {
        cuda_push_array(l.biases_gpu, base.biases, l.outputs);
        cuda_push_array(l.weights_gpu, base.weights, l.outputs*l.inputs);
    }
}

void sync_layer(network **nets, int n, int j)
{
    int i;
    network *net = nets[0];
    layer base = net->layers[j];
    scale_weights(base, 0);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        pull_weights(l);
        merge_weights(l, base);
    }
    scale_weights(base, 1./n);
    for (i = 0; i < n; ++i) {
        cuda_set_device(nets[i]->gpu_index);
        layer l = nets[i]->layers[j];
        distribute_weights(l, base);
    }
}

typedef struct{
    network **nets;
    int n;
    int j;
} sync_args;

void *sync_layer_thread(void *ptr)
{
    sync_args args = *(sync_args*)ptr;
    sync_layer(args.nets, args.n, args.j);
    free(ptr);
    return 0;
}

pthread_t sync_layer_in_thread(network **nets, int n, int j)
{
    pthread_t thread;
    sync_args *ptr = (sync_args *)calloc(1, sizeof(sync_args));
    ptr->nets = nets;
    ptr->n = n;
    ptr->j = j;
    if(pthread_create(&thread, 0, sync_layer_thread, ptr)) error("Thread creation failed");
    return thread;
}

void sync_nets(network **nets, int n, int interval)
{
    int j;
    int layers = nets[0]->n;
    pthread_t *threads = (pthread_t *) calloc(layers, sizeof(pthread_t));

    *(nets[0]->seen) += interval * (n-1) * nets[0]->batch * nets[0]->subdivisions;
    for (j = 0; j < n; ++j){
        *(nets[j]->seen) = *(nets[0]->seen);
    }
    for (j = 0; j < layers; ++j) {
        threads[j] = sync_layer_in_thread(nets, n, j);
    }
    for (j = 0; j < layers; ++j) {
        pthread_join(threads[j], 0);
    }
    free(threads);
}

float train_networks(network **nets, int n, data d, int interval)
{
    int i, j;
    int batch = nets[0]->batch;
    int n_loss = nets[0]->n_loss;

    int subdivisions = nets[0]->subdivisions;
    assert(batch * subdivisions * n == d.X.rows);
    pthread_t *threads = (pthread_t *) calloc(n, sizeof(pthread_t));
    float *errors = (float *) calloc(n, sizeof(float));

    float sum = 0;
    for(i = 0; i < n; ++i){
        data p = get_data_part(d, i, n);
        threads[i] = train_network_in_thread(nets[i], p, errors + i);
    }
    for(i = 0; i < n; ++i){
        pthread_join(threads[i], 0);
        sum += errors[i];
        for (j = 1; j < n_loss; ++j) {
            nets[0]->sub_loss[j] += nets[i]->sub_loss[j];
        }
    }
    for(j = 0; j < n_loss; ++j) {
        nets[0]->sub_loss[j] /= n;
    }

    if (get_current_batch(nets[0]) % interval == 0) {
        printf("Syncing... ");
        fflush(stdout);
        sync_nets(nets, n, interval);
        printf("Done!\n");
    }
    free(threads);
    free(errors);
    return (float)sum/(n);
}

void pull_network_output(network *net)
{
    layer l = get_network_output_layer(net);
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
}

void pre_transform_conv_params(network *net) {
    printf("Perform pre-transform for convolutional layers\n");
    int nl = net->n;
    int i;
    for (i = 0; i < nl; i++) {
        layer l = net->layers[i];
        if (strcmp(get_layer_string(l.type),"convolutional")) continue;
        transform_weights(l.weights_gpu, l.n, l.c/l.groups*l.size*l.size, l.tran_weights_gpu, l.weight_transform, l.q_coeff_gpu, l.n_coeff);
    }
}

void swap_weight_transform(layer *l){
    float *swap = l->weights;
    l->weights = l->tran_weights;
    l->tran_weights = swap;

    swap = l->weights_gpu;
    l->weights_gpu = l->tran_weights_gpu;
    l->tran_weights_gpu = swap;
}

void save_training_info(FILE* file, network* net, int header, int N) {
    if (file == NULL) {
        return;
    }
    if (header) {
        int current_batch = get_current_batch(net) + 1;
        int batch_size = net->batch;
        int subdiv = net->subdivisions;
        fwrite(&N, sizeof(int), 1, file);
        fwrite(&batch_size, sizeof(int), 1, file);
        fwrite(&subdiv, sizeof(int), 1, file);
        fwrite(&current_batch, sizeof(int), 1, file);
        fwrite(&(net->n_loss), sizeof(int), 1, file);
        fwrite(net->sub_loss_id, sizeof(int), net->n_loss, file);
    } else {
        fwrite(net->sub_loss, sizeof(float), net->n_loss, file);
    }
    return;
}