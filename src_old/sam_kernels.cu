#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "sam_layer.h"
#include "cuda.h"
#include "blas.h"
}

extern "C" void forward_sam_layer_gpu(layer l, network net)
{
    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = 1;

    sam_gpu(l.output_gpu, size, channel_size, net.input_gpu, l.output_gpu);

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

extern "C" void backward_sam_layer_gpu(layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = 1;

    backward_sam_gpu(l.delta_gpu, size, channel_size, net.input_gpu, l.delta_gpu, l.output_gpu, net.delta_gpu);
}