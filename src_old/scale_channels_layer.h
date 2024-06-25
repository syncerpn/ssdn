#ifndef SCALE_CHANNELS_LAYER_H
#define SCALE_CHANNELS_LAYER_H

#include "layer.h"
#include "network.h"

layer make_scale_channels_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2, int scale_wh);
void resize_scale_channels_layer(layer *l, network *net);

void forward_scale_channels_layer_gpu(layer l, network net);
void backward_scale_channels_layer_gpu(layer l, network net);

#endif  // SCALE_CHANNELS_LAYER_H