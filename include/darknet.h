#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#include "cudnn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;
extern unsigned int seed; //nghiant_20190820: add seed support for exact result reproduction

typedef struct{
    int classes;
    char **names;
} metadata;

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU,
    ABLEAKY, SWISH, MISH
} ACTIVATION;

typedef struct{
    ACTIVATION type;
    float alpha;
    float beta;
} activation_scheme;

typedef enum{
    NONE = 0, XAVIER, HE, NORMAL, SCALAR, UNIFORM
} INITIALIZER;

typedef enum{
    NMS_NORMAL = 0, NMS_SOFT_L, NMS_SOFT_G
} NMS_MODE;

typedef struct{
    INITIALIZER type;
    float alpha;
    float mu;
    float sigma;
    float auto_sigma;
} initializer;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum {
    NETWORK = -1,
    BLANK = 0,
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    QUANTIZATION,
    DIFF,
    STACK,
    SR_FLAT,
    NORM_W_CONVOLUTIONAL,
    SAM,
    SCALE_CHANNELS,
    CUSTOM_CONV,
} LAYER_TYPE;

typedef enum{
    WTS_NONE = 0,
    WTS_MAX_SHIFTER,
    WTS_MEAN_SHIFTER,
    WTS_UNIFORM,
    WTS_SHIFTER,
    WTS_SCALE_LINEAR,
} WEIGHT_TRANSFORM_SCHEME;

typedef struct weight_transform_scheme{
    WEIGHT_TRANSFORM_SCHEME type;
    float step_size;
    int num_level;
    int first_shifting_factor;
} weight_transform_scheme;

typedef enum{
    QS_NONE = 0,
    QS_UNIFORM,
    QS_ROOT,
    QS_BIT
} QUANTIZATION_SCHEME;

typedef struct{
    QUANTIZATION_SCHEME type;
    float step_size;
    int num_level;
    float root;
    int zero_center;
} quantization_scheme;
//nghiant_end

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH, WGAN, SYMEXP, LOGCOSH
} COST_TYPE;

typedef enum {
    SGD, ADAM
    //NAG, ADAGRAD, ADADELTA, RMSPROP, ADAMAX, AMSGRAD, NADAM
} OPTIMIZER;

typedef struct{
    int max_w;
    int max_h;
    float prob;
} cutout_args;

typedef struct{
    OPTIMIZER optim;
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct network;
typedef struct network network;

typedef struct layer{
    unsigned int self_index;
    LAYER_TYPE type;
    activation_scheme activation;

    initializer initializer;

    quantization_scheme quantization;
    weight_transform_scheme weight_transform;

    COST_TYPE cost_type;
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);

    char name[64];
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    //nghiant
    int dilation;
    //global: impact
    float impact; //layer impact factor
    //yolo and region warmup param
    int warmup;
    int blind;
    int logistic_derivative; //yolo correct derivative backward
    //nghiant_end
    int index;
    int steps;
    int hidden;
    int truth;
    float smooth;

    float angle;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;

    int noadjust;
    int reorg;
    int tanh;

    int *mask;
    int *mask_gpu;
    //tree
    int *skip_index;
    int *skip_index_gpu;
    //tree_end
    int total;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;
    int bias_match;
    float ignore_thresh;
    float truth_thresh;

    float thresh;
    // float focus;

    int onlyforward;
    int stopbackward;
    int frozen;
    int root_connect;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    int* indexes;
    int* input_layers;
    int* input_sizes;
    int* map;

    int* map_gpu;

    int* counts;
    float** sums;
    float* rand;
    float* cost;
    float* state;
    float* prev_state;
    float* forgot_state;
    float* forgot_delta;
    float* state_delta;
    float* combine_cpu;
    float* combine_delta_cpu;

    float* concat;
    float* concat_delta;

    float* tran_weights;

    float* biases;
    float* bias_updates;

    float* scales;
    float* scale_updates;

    float* weights;
    float* weight_updates;

    float* delta;
    float* output;
    float* loss;
    float* squared;
    float* norms;
    float* norms_delta;
    float* norms_delta_gpu;

    float* spatial_mean;
    float* mean;
    float* variance;

    float* mean_delta;
    float* variance_delta;

    float* rolling_mean;
    float* rolling_variance;

    float* x;
    float* x_norm;

    float* m;
    float* v;
    
    float* bias_m;
    float* bias_v;
    float* scale_m;
    float* scale_v;


    float* z_cpu;
    float* r_cpu;
    float* h_cpu;
    float* prev_state_cpu;

    float* temp_cpu;
    float* temp2_cpu;
    float* temp3_cpu;

    float* dh_cpu;
    float* hh_cpu;
    float* prev_cell_cpu;
    float* cell_cpu;
    float* f_cpu;
    float* i_cpu;
    float* g_cpu;
    float* o_cpu;
    float* c_cpu;
    float* dc_cpu;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    float * tran_weights_gpu;
    int n_coeff;
    float * q_coeff;
    float * q_coeff_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;

    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;

    float * input_im;
    float * input_im_gpu;

    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;

    //nghiant_20200221: diff layer option
    int mask_layer_softmax;

    //nghiant_20201102: norm_w_convolutional layer option
    int nweights_v;
    int nweights_g;

    float* weights_v;
    float* weights_g;
    float* weights_v_gpu;
    float* weights_g_gpu;

    float* weight_updates_v;
    float* weight_updates_g;
    float* weight_updates_v_gpu;
    float* weight_updates_g_gpu;

    float* weights_v_norm;
    float* weights_v_norm_gpu;

    float sr_flat_scale;

    int scale_wh;
    
} layer;

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM,
    CYCLICAL, PULSE
} LEARNING_RATE_POLICY;

typedef struct network{
    //nghiant
    initializer initializer;
    int pre_transformable;
    int pre_transform;
    float lower_bound;
    float upper_bound;
    int n_loss; //report multiple loss terms separately; mostly used to know how good knowledge distillation is.
    float* sub_loss;
    int* sub_loss_id;
    //nghiant_end
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    layer *layers;
    float *output;
    LEARNING_RATE_POLICY policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    OPTIMIZER optim;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    //nghiant
    float jitter; //moved from layer; dont know why authors put this in layer (?!)
    float* zoom;
    float hflip;
    float vflip;
    float solarize;
    float posterize;
    float noise;
    cutout_args cutout;
    //nghiant_end
    int random;

    int gpu_index;
    tree *hierarchy;

    float *input;
    float *truth;
    float *delta;

    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;

    float *output_gpu;

    float total_CONV_w_size;
} network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct {
    char id[128];
    box bbox;
} im_box;

typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
    int* random_indices;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, IMAGE_DATA, IMAGE_DATA_CROP, OLD_CLASSIFICATION_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA,
    LETTERBOX_DATA_8BIT, LETTERBOX_DATA_NO_TRUTH,
    AUTO_COLORIZE_DATA,
    LP_SUPERRES_DATA,
    LP_SUPERRES_TF_DATA,
    LP_SUPERRES_TF_LESS_DATA,
} DATA_TYPE;

typedef struct load_args{
    int threads;
    
    //nghiant_20191107
    int thread_id;
    int* random_array;
    //nghiant_20191107_end

    int us_factor;
    float **all_data_im;
    float **all_data_gt;

    char *train_directory;
    char *im_prefix;
    char *gt_prefix;
    
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;

    float* zoom;
    float hflip;
    float vflip;
    float solarize;
    float posterize;
    float noise;
    cutout_args cutout;

    data *d;
    image *im;
    image *resized;
    DATA_TYPE type;
    tree *hierarchy;

    char* label_dir;
    float* truth;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

//layer.c
void free_layer(layer);

//tree.c
tree *read_tree(char *filename);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

//data.c
pthread_t load_data(load_args args);
pthread_t load_data_in_thread(load_args args);
char **get_labels(char *filename);
char **get_labels_with_n(char *filename, int* n);
void get_next_batch(data d, int n, int offset, float *X, float *y);
void summarize_data_augmentation_options(load_args args);
void free_data(data d);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
box_label *read_boxes(char *filename, int *n);
list *get_paths(char *filename);

//option_list.c
char *option_find_str(list *l, char *key, char *def);
char *option_find_str_quiet(list *l, char *key, char *def);
void option_find_str_series(list *l, char *key, int* num, char*** series);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);
void option_find_int_series(list *l, char *key, int* num, int** series);
void option_find_float_series(list *l, char *key, int* num, float** series);
list *read_module_cfg(char *filename);
list *read_data_cfg(char *filename);
metadata get_metadata(char *file);

//parser.c
void save_convolutional_weights(layer l, FILE *fp);
void save_batchnorm_weights(layer l, FILE *fp);
void save_connected_weights(layer l, FILE *fp);
void save_weights(network *net, char *filename);
void load_modular_weights(network* net, char* modules);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);
void print_layer(layer l);
void print_optimizer(network* net);
network *parse_network_cfg(char *filename, int batch);
list *read_cfg(char *filename);

//utils.c
double what_time_is_it_now();
unsigned char *read_file(char *filename);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *find_char_2arg(int argc, char **argv, char *arg1, char *arg2, char *def);
char *basecfg(char *cfgfile);
char *fgetl(FILE *fp);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
void merge(float* arr, int l, int m, int r, int* sorted_index);
void merge_sort(float* arr, int l, int r, int* sorted_index);
void scale_array(float *a, int n, float s);
void strip(char *s);
void top_k(float *a, int n, int k, int *index);
void clear_prev_line(FILE* stream);
void clear_n_prev_lines(FILE* stream, int n);
void error(const char *s);
void normalize_array(float *a, int n);

int *read_map(char *filename);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
int find_int_arg(int argc, char **argv, char *arg, int def);
int find_int_2arg(int argc, char **argv, char *arg1, char *arg2, int def);
int find_arg(int argc, char* argv[], char *arg);
int find_2arg(int argc, char* argv[], char *arg1, char *arg2);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
float sec(clock_t clocks);
float rand_normal();
float rand_uniform(float min, float max);
float find_float_arg(int argc, char **argv, char *arg, float def);
float find_float_2arg(int argc, char **argv, char *arg1, char* arg2, float def);
size_t rand_size_t();

//network.c
void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
void pre_transform_conv_params(network *net);
void swap_weight_transform(layer *l);
void free_network(network *net);
void top_predictions(network *net, int n, int *index);
void save_training_info(FILE* file, network* net, int header, int N);
void visualize_network(network *net);
void copy_detection(detection* dst, detection* src, int n_classes, size_t nboxes);
void free_detections(detection *dets, int n);
void reset_network_state(network *net, int b);
float *network_accuracies(network *net, data d, int n);
float train_network_datum(network *net);
float train_networks(network **nets, int n, data d, int interval);
float train_network(network *net, data d);

float *network_predict(network *net, float *input);
float get_current_rate(network *net);
char *get_layer_string(LAYER_TYPE a);
int resize_network(network *net, int w, int h);
size_t get_current_batch(network *net);
layer get_network_output_layer(network *net);
matrix network_predict_data(network *net, data test);
image get_network_image_layer(network *net, int i);
image get_network_image(network *net);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
network *load_network(char *cfg, char *weights, char* modules, int clear, int batch);
load_args get_base_args(network *net);

//blas.c
float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

//image.c
int best_3d_shift_r(image a, image b, int min, int max);
int show_image(image p, const char *name, int ms);
void draw_label(image a, int r, int c, image label, const float *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void grayscale_image_3c(image im);
void normalize_image(image p);
void rgbgr_image(image im);
void censor_image(image im, int dx, int dy, int w, int h);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
void composite_3d(char *f1, char *f2, char *out, int delta);
void constrain_image(image im);
void find_min_max(float a, float b, float c, float d, float* min, float* max);
void solarize_image(image im, float threshold);
void posterize_image(image im, int levels);
void random_distort_image_extend(image im, float solarize, float posterize, float noise);

void flip_image_x(image a, int h_flip, int v_flip);
void flip_image_horizontal(image a);
void flip_image_vertical(image a);
void ghost_image(image source, image dest, int dx, int dy);
void random_distort_image(image im, float hue, float saturation, float exposure);
void random_cutout_image(image im, cutout_args cutout);

void fill_image(image m, float s);
void rotate_image_cw(image im, int times);
void draw_detection(image im, im_box ib, float red, float green, float blue);
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);
void free_image(image m);
image get_label(image **characters, char *string, int size);
image make_random_image(int w, int h, int c);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, float thresh);
image mask_to_rgb(image mask);
image copy_image(image p);
image rotate_image(image m, float rad);
image rotate_image_preserved(image im, float rad);
image float_to_image(int w, int h, int c, float *data);
image grayscale_image(image im);
image real_to_heat_image(float* data, int w, int h);
image **load_alphabet();

//image_kernels.cu
void flip_image_x_gpu(float* im_data, int w, int h, int c, int h_flip, int v_flip);
void random_distort_image_gpu(float* im_data, int w, int h, int c, float hue, float saturation, float exposure);
void random_distort_image_extend_gpu(float* im_data, int w, int h, int c, float solarize, float posterize, float noise);
void random_cutout_image_gpu(float* im_data, int w, int h, int c, cutout_args cutout);
void resize_image_gpu(float* input, int iw, int ih, float* output, int ow, int oh, int oc);

//blas_kernels.cu
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void fill_int_gpu(int N, int ALPHA, int * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);
void floorf_gpu(int N, float * X, int INCX);

//cuda.c
int *cuda_make_int_array(int *x, size_t n);
void cuda_set_device(int n);
void cuda_free(float *x_gpu);
void cuda_free_int(int *x_gpu);
void cuda_push_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_push_int_array(int *x_gpu, int *x, size_t n);
void cuda_pull_int_array(int *x_gpu, int *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
float* cuda_make_array(float *x, size_t n);

//matrix.c
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_to_csv(matrix m);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
void free_matrix(matrix m);
matrix make_matrix(int rows, int cols);
matrix csv_to_matrix(char *filename);

//connected_layer.c
void denormalize_connected_layer(layer l);
void statistics_connected_layer(layer l);

//convolutional_layer.c
void denormalize_convolutional_layer(layer l);
void rescale_weights(layer l, float scale, float trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

//demo.c
void demo(char *cfgfile, char *weightfile, char* modules, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen, int cropx, int save_frame, int add_frame_count, char* line_150);
void demo_mf(char *cfgfile, char *weightfile, char* modules, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, int cropx, int save_frame, int add_frame_count);
int size_network(network *net);

//detection_layer.c
void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

//yolo_layer.c
void zero_objectness(layer l);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);

//region_layer.c
void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);

//box.c
float box_iou(box a, box b);
box float_to_box(float *f, int stride);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh, NMS_MODE mode);

//image_opencv.cpp
image get_image_from_stream(void *p);
void *open_video_stream(const char *f, int c, int w, int h, int fps);
void make_window(char *name, int w, int h, int fullscreen);

//list.c
void **list_to_array(list *l);
void free_list(list *l);


#ifdef __cplusplus
}
#endif
#endif
