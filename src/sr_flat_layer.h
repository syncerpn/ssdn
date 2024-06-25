#ifndef SR_FLAT_LAYER_H
#define SR_FLAT_LAYER_H
#include "darknet.h"

layer make_sr_flat_layer(int batch, int w, int h, int c, float scale);
void resize_sr_flat_layer(layer *l, int w, int h);

void forward_sr_flat_layer_gpu(const layer l, network net);
void backward_sr_flat_layer_gpu(const layer l, network net);

#endif