#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"

void pull_network_output(network *net);
char *get_layer_string(LAYER_TYPE a);
network *make_network(int n);
int resize_network(network *net, int w, int h);
void calc_network_cost(network *net);

#endif