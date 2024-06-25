#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

image get_crop_image(layer l);
layer make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle, float saturation, float exposure);
void resize_crop_layer(layer *l, int w, int h);
void forward_crop_layer_gpu(layer l, network net);
void backward_crop_layer_gpu(const layer l, network net);

#endif