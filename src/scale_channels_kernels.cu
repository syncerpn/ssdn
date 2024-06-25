#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "activations.h"
#include "scale_channels_layer.h"
#include "cuda.h"
#include "blas.h"
}

void forward_scale_channels_layer_gpu(layer l, network net)
{
    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = l.out_w * l.out_h;
    int batch_size = l.out_c * l.out_w * l.out_h;

    scale_channels_gpu(net.layers[l.index].output_gpu, size, channel_size, batch_size, l.scale_wh, net.input_gpu, l.output_gpu);

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_scale_channels_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = l.out_w * l.out_h;
    int batch_size = l.out_c * l.out_w * l.out_h;
    float *from_output = net.layers[l.index].output_gpu;
    float *from_delta = net.layers[l.index].delta_gpu;

    backward_scale_channels_gpu(l.delta_gpu, size, channel_size, batch_size, l.scale_wh, net.input_gpu, from_delta, from_output, net.delta_gpu);
}