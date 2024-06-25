#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes, float* anchors, int softmax, int background);
void resize_yolo_layer(layer *l, int w, int h);
int yolo_num_detections(layer l, float thresh);

void forward_yolo_layer_gpu(const layer l, network net);
void backward_yolo_layer_gpu(layer l, network net);

#endif