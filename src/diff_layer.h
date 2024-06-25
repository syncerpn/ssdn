#ifndef DIFF_LAYER_H
#define DIFF_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_diff_layer(network* net, int batch, int truth_layer, int learn_layer, COST_TYPE cost_type, int* skip_list, int skip_list_length);
void resize_diff_layer(layer *l, network* net);
void forward_diff_layer_gpu(const layer l, network net);
void backward_diff_layer_gpu(layer l, network net);

#endif