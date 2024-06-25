#ifndef SAM_LAYER_H
#define SAM_LAYER_H

#include "layer.h"
#include "network.h"
#include "activations.h"

layer make_sam_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);

void resize_sam_layer(layer *l, int w, int h);
void forward_sam_layer_gpu(const layer l, network net);
void backward_sam_layer_gpu(const layer l, network net);

#endif  // SAM_LAYER_H