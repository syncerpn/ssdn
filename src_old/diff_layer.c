#include "diff_layer.h"
#include "cost_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_diff_layer(network* net, int batch, int truth_layer, int learn_layer, COST_TYPE cost_type, int* skip_list, int skip_list_length) {
	layer l = {0};
	l.type = DIFF;
	l.cost = calloc(1, sizeof(float));
    l.cost_type = cost_type;
	l.batch = batch;

	l.input_layers = calloc(2, sizeof(int));
	l.input_layers[0] = truth_layer;
	l.input_layers[1] = learn_layer;

	layer lt = net->layers[truth_layer];
	layer ll = net->layers[learn_layer];
	assert(lt.out_w == ll.out_w);
	assert(lt.out_h == ll.out_h);
	assert(lt.out_c == ll.out_c);

	l.out_w = ll.out_w;
	l.out_h = ll.out_h;
	l.out_c = ll.out_c;
	l.outputs = l.out_w * l.out_h * l.out_c;

    l.output = calloc(batch * l.outputs, sizeof(float));
    l.delta = calloc(batch * l.outputs, sizeof(float));

    l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);

    l.w = l.out_w;
	l.h = l.out_h;
	l.c = l.out_c;
	l.inputs = l.outputs;

	l.n = skip_list_length;
	if (l.n > 0) {
		l.indexes = calloc(l.n, sizeof(int));
		memcpy(l.indexes, skip_list, l.n * sizeof(int));
		l.indexes_gpu = cuda_make_int_array(l.indexes, l.n);
	}
	
    l.forward_gpu = forward_diff_layer_gpu;
    l.backward_gpu = backward_diff_layer_gpu;

    return l;
}

void resize_diff_layer(layer *l, network* net) {
	layer lt = net->layers[l->input_layers[0]];

	l->out_w = lt.out_w;
	l->out_h = lt.out_h;
	l->out_c = lt.out_c;
	l->outputs = l->out_w * l->out_h * l->out_c;
	
	l->w = l->out_w;
	l->h = l->out_h;
	l->c = l->out_c;
	l->inputs = l->outputs;
	
    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta  = realloc(l->delta,  l->batch*l->outputs*sizeof(float));

    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
}