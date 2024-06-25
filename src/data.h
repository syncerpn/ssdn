#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#include "darknet.h"
#include "matrix.h"
#include "list.h"
#include "image.h"
#include "tree.h"

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout);
data load_data_detection(int n, char **paths, char* label_dir, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure, float angle, float* zoom, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout);
data load_data_letterbox_no_truth(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure, float angle, float* zoom, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout);
data load_data_regression(char **paths, int n, int m, int classes, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout);
data load_data_lp_superres(int n, char** paths, int m, int w, int h, int us_factor);
data load_data_lp_superres_tf(int n, float** all_data_im, float** all_data_gt, int m, int w, int h, int us_factor);
data load_data_lp_superres_tf_less(int n, char *train_directory, char *im_prefix, char *gt_prefix, int m, int w, int h, int us_factor);

data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
data concat_datas(data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);
void get_random_batch(data d, int n, float *X, float *y);

#endif
