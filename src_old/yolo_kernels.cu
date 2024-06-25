#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "activations.h"
#include "yolo_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "box.h"
}

__device__ float overlap_y(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

__device__ float box_intersection_y(float px, float py, float pw, float ph, float tx, float ty, float tw, float th)
{
    float w = overlap_y(px, pw, tx, tw);
    float h = overlap_y(py, ph, ty, th);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

__device__ float box_union_y(float px, float py, float pw, float ph, float tx, float ty, float tw, float th)
{
    float i = box_intersection_y(px, py, pw, ph, tx, ty, tw, th);
    float u = pw*ph + tw*th - i;
    return u;
}

__device__ float box_iou_kernel_y(float px, float py, float pw, float ph, float tx, float ty, float tw, float th) {
    return box_intersection_y(px, py, pw, ph, tx, ty, tw, th) / box_union_y(px, py, pw, ph, tx, ty, tw, th);
}

__global__ void delta_yolo_a_kernel(float* output, float* delta, float* biases, int* mask, float* truth, int ltruths, int* skip_index,
    int batch, int lw, int lh, int ln, int max_boxes, int classes, int netw, int neth, float ignore_thresh, float truth_thresh, int* map,
    float class_scale, float coord_scale, float noobject_scale, float object_scale, int softmax, int background, int warmup, int netseen) {

    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= batch * lh * lw * ln) return;

    int loutputs = lw * lh * ln * (classes + 5);
    int n = index % ln;
    index = (index - n) / ln;
    int i = index % lw;
    index = (index - i) / lw;
    int j = index % lh;
    int b = (index - j) / lh;
    int t,k;
    if (skip_index[b]) return;

    int box_index = b * loutputs + n * lw * lh * (classes + 5) + 0 * lw * lh + j * lw + i;
    int obj_index = b * loutputs + n * lw * lh * (classes + 5) + 4 * lw * lh + j * lw + i;
    int cls_index = b * loutputs + n * lw * lh * (classes + 5) + 5 * lw * lh + j * lw + i;

    float px = (i + output[box_index + 0 * lw * lh]) / lw;
    float py = (j + output[box_index + 1 * lw * lh]) / lh;
    float pw = exp(output[box_index + 2 * lw * lh]) * biases[2 * mask[n]] / netw;
    float ph = exp(output[box_index + 3 * lw * lh]) * biases[2 * mask[n] + 1] / neth;
    
    float best_iou = 0;
    int best_t = 0;
    for (t = 0; t < max_boxes; ++t) {
        float tx = truth[t * 5 + b * ltruths + 0];
        float ty = truth[t * 5 + b * ltruths + 1];
        float tw = truth[t * 5 + b * ltruths + 2];
        float th = truth[t * 5 + b * ltruths + 3];
        
        if (tx == 0) break;

        float iou = box_iou_kernel_y(px, py, pw, ph, tx, ty, tw, th);
        if (iou > best_iou) {
            best_iou = iou;
            best_t = t;
        }
    }

    delta[obj_index] = noobject_scale * (0 - output[obj_index]);

    if (background) delta[obj_index] = noobject_scale * (1 - output[obj_index]);

    if (best_iou > ignore_thresh) delta[obj_index] = 0;

    if (best_iou > truth_thresh) {
        delta[obj_index] = object_scale * (1 - output[obj_index]);
        
        int truth_class = truth[best_t * 5 + b * ltruths + 4];

        if (map) truth_class = map[truth_class];

        if (delta[cls_index] && !softmax) {
            delta[cls_index + lw * lh * truth_class] = class_scale * (1 - output[cls_index + lw * lh * truth_class]);

        } else {
            for (k = 0; k < classes; ++k) {
                delta[cls_index + lw * lh * k] = class_scale * (((k == truth_class) ? 1 : 0) - output[cls_index + lw * lh * k]);                
            }
        }

        float tx = truth[best_t * 5 + b * ltruths + 0];
        float ty = truth[best_t * 5 + b * ltruths + 1];
        float tw = truth[best_t * 5 + b * ltruths + 2];
        float th = truth[best_t * 5 + b * ltruths + 3];

        float scale = coord_scale * (2 - tw * th);

        tx = tx * lw - i;
        ty = ty * lh - j;
        tw = log(tw * netw / biases[2 * mask[n]]);
        th = log(th * neth / biases[2 * mask[n] + 1]);

        delta[box_index + 0 * lw * lh] = scale * (tx - output[box_index + 0 * lw * lh]);
        delta[box_index + 1 * lw * lh] = scale * (ty - output[box_index + 1 * lw * lh]);
        delta[box_index + 2 * lw * lh] = scale * (tw - output[box_index + 2 * lw * lh]);
        delta[box_index + 3 * lw * lh] = scale * (th - output[box_index + 3 * lw * lh]);
    }
    //warmup: need to check
    if (netseen < warmup) {
        float tx = (i + 0.5) / lw;
        float ty = (j + 0.5) / lh;
        float tw = biases[2*n]   / netw;
        float th = biases[2*n+1] / neth;

        float scale = 0.01;

        tx = tx * lw - i;
        ty = ty * lh - j;
        tw = log(tw * netw / biases[2 * n]);
        th = log(th * neth / biases[2 * n + 1]);

        delta[box_index + 0 * lw * lh] = scale * (tx - output[box_index + 0 * lw * lh]);
        delta[box_index + 1 * lw * lh] = scale * (ty - output[box_index + 1 * lw * lh]);
        delta[box_index + 2 * lw * lh] = scale * (tw - output[box_index + 2 * lw * lh]);
        delta[box_index + 3 * lw * lh] = scale * (th - output[box_index + 3 * lw * lh]);
    }
    //warmup_end
}

void delta_yolo_a_gpu(float* output, float* delta, float* biases, int* mask, float* truth, int ltruths, int* skip_index,
    int batch, int lw, int lh, int ln, int max_boxes, int classes, int netw, int neth, float ignore_thresh, float truth_thresh, int* map,
    float class_scale, float coord_scale, float noobject_scale, float object_scale, int softmax, int background, int warmup, int netseen) {

    delta_yolo_a_kernel<<<cuda_gridsize(batch * lh * lw * ln), BLOCK>>>(output, delta, biases, mask, truth, ltruths, skip_index, batch, lw, lh, ln, max_boxes, classes, netw, neth, ignore_thresh, truth_thresh, map, class_scale, coord_scale, noobject_scale, object_scale, softmax, background, warmup, netseen);
    check_error(cudaPeekAtLastError());
}

__global__ void delta_yolo_b_kernel(float* output, float* delta, float* biases, int* mask, float* truth, int ltruths, int* skip_index, int ltotal,
    int batch, int lw, int lh, int ln, int max_boxes, int classes, int netw, int neth, int* map,
    float class_scale, float coord_scale, float object_scale, int softmax, int background, int rescore, int bias_match) {

    int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= batch) return;
    if (skip_index[index]) return;

    int loutputs = lw * lh * ln * (classes + 5);
    int b = index;
    int t, k, n, nm;
    int i, j;
    float px, py, pw, ph, tx, ty, tw, th;
    float iou, best_iou;
    int best_n, mask_n;
    int box_index, obj_index, cls_index;
    float scale;
    int truth_class;

    for (t = 0; t < max_boxes; ++t) {

        tx = truth[t * 5 + b * ltruths + 0];
        ty = truth[t * 5 + b * ltruths + 1];
        tw = truth[t * 5 + b * ltruths + 2];
        th = truth[t * 5 + b * ltruths + 3];

        if (tx == 0) break;

        best_iou = 0;
        best_n = 0;

        i = (tx * lw);
        j = (ty * lh);

        for (n = 0; n < ltotal; ++n) {

            px = 0;
            py = 0;
            pw = biases[2*n] / netw;
            ph = biases[2*n + 1] / neth;

            if (!bias_match) {
                for (nm = 0; nm < ln; ++nm) {
                    if (n == mask[nm]) {
                        int nn  = (nm * lw * lh + j * lw + i) / (lw * lh);
                        int loc = (nm * lw * lh + j * lw + i) % (lw * lh);

                        box_index = b * loutputs + nn * lw * lh * (classes + 5) + 0 * lw * lh + loc;
                        pw = exp(output[box_index + 2 * lw * lh]) * biases[2*n] / netw;
                        ph = exp(output[box_index + 3 * lw * lh]) * biases[2*n+1] / neth;
                    }
                }
            }

            iou = box_iou_kernel_y(px, py, pw, ph, 0, 0, tw, th);

            if (iou > best_iou) {
                best_iou = iou;
                best_n = n;
            }
        }

        mask_n = -1;
        for (n = 0; n < ln; ++n) {
            if (mask[n] == best_n) {
                mask_n = n;
                break;
            }
        }
        if (mask_n >= 0) {

            box_index = b * loutputs + mask_n * lw * lh * (classes + 5) + 0 * lw * lh + j * lw + i;
            obj_index = b * loutputs + mask_n * lw * lh * (classes + 5) + 4 * lw * lh + j * lw + i;
            cls_index = b * loutputs + mask_n * lw * lh * (classes + 5) + 5 * lw * lh + j * lw + i;

            px = (i + output[box_index + 0 * lw * lh]) / lw;
            py = (j + output[box_index + 1 * lw * lh]) / lh;
            pw = exp(output[box_index + 2 * lw * lh]) * biases[2*best_n] / netw;
            ph = exp(output[box_index + 3 * lw * lh]) * biases[2*best_n+1] / neth;

            iou = box_iou_kernel_y(px, py, pw, ph, tx, ty, tw, th);

            scale = coord_scale * (2 - tw * th);

            tx = tx * lw - i;
            ty = ty * lh - j;
            tw = log(tw * netw / biases[2 * best_n]);
            th = log(th * neth / biases[2 * best_n + 1]);

            delta[box_index + 0 * lw * lh] = scale * (tx - output[box_index + 0 * lw * lh]);
            delta[box_index + 1 * lw * lh] = scale * (ty - output[box_index + 1 * lw * lh]);
            delta[box_index + 2 * lw * lh] = scale * (tw - output[box_index + 2 * lw * lh]);
            delta[box_index + 3 * lw * lh] = scale * (th - output[box_index + 3 * lw * lh]);

            delta[obj_index] = object_scale * (1 - output[obj_index]);

            if (rescore) delta[obj_index] = object_scale * (iou - output[obj_index]);

            if (background) delta[obj_index] = object_scale * (0 - output[obj_index]);

            truth_class = truth[t * 5 + b * ltruths + 4];
            if (map) truth_class = map[truth_class];
            if (delta[cls_index] && !softmax) {
                delta[cls_index + lw * lh * truth_class] = class_scale * (1 - output[cls_index + lw * lh * truth_class]);
                
            } else {
                for (k = 0; k < classes; ++k) {
                    delta[cls_index + lw * lh * k] = class_scale * (((k == truth_class) ? 1 : 0) - output[cls_index + lw * lh * k]);
                }
            }
        }
    }
}

void delta_yolo_b_gpu(float* output, float* delta, float* biases, int* mask, float* truth, int ltruths, int* skip_index, int ltotal,
    int batch, int lw, int lh, int ln, int max_boxes, int classes, int netw, int neth, int* map,
    float class_scale, float coord_scale, float object_scale, int softmax, int background, int rescore, int bias_match) {

    delta_yolo_b_kernel<<<cuda_gridsize(batch), BLOCK>>>(output, delta, biases, mask, truth, ltruths, skip_index, ltotal, batch, lw, lh, ln, max_boxes, classes, netw, neth, map, class_scale, coord_scale, object_scale, softmax, background, rescore, bias_match);
    check_error(cudaPeekAtLastError());
}

void delta_yolo_class(float *output, float *delta, int index, int class_id, int classes, tree *hier, float scale, int stride, int tag)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class_id >= 0){
            pred *= output[index + stride*class_id];
            int g = hier->group[class_id];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + stride*(offset + i)] = scale * (0 - output[index + stride*(offset + i)]);
            }
            delta[index + stride*class_id] = scale * (1 - output[index + stride*class_id]);

            class_id = hier->parent[class_id];
        }
    } else {
        if (delta[index] && tag){
            delta[index + stride*class_id] = scale * (1 - output[index + stride*class_id]);
            return;
        }

        for(n = 0; n < classes; ++n){
            delta[index + stride*n] = scale * (((n == class_id)?1 : 0) - output[index + stride*n]);
        }
    }
}

int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

extern "C" void forward_yolo_layer_gpu(const layer l, network net) {
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);

    fill_int_gpu(l.batch, 0, l.skip_index_gpu, 1);

    int b, n, t;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = b * l.outputs + n * l.w * l.h * (l.classes + 5) + 0 * l.w * l.h;
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, (activation_scheme){LOGISTIC});
            index = b * l.outputs + n * l.w * l.h * (l.classes + 5) + 4 * l.w * l.h;
            if (!l.background) activate_array_gpu(l.output_gpu + index, l.w*l.h, (activation_scheme){LOGISTIC});
            index = b * l.outputs + n * l.w * l.h * (l.classes + 5) + 5 * l.w * l.h;
            if (!l.softmax && !l.softmax_tree) activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, (activation_scheme){LOGISTIC});
        }
    }
    if (l.softmax_tree) {
        int index = 5 * l.w * l.h;
        softmax_tree(net.input_gpu + index, l.w * l.h, l.batch * l.n, l.inputs/l.n, 1, l.output_gpu + index, *l.softmax_tree);
    } else if (l.softmax) {
        int index = (4 + !l.background) * l.w * l.h;
        softmax_gpu(net.input_gpu + index, l.classes + l.background, l.batch*l.n, l.inputs/l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }
    
    *(l.cost) = 0;

    if(l.softmax_tree){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        for (b = 0; b < l.batch; ++b) {
            l.skip_index[b] = 0;
            for(t = 0; t < l.max_boxes; ++t){
                box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                if(!truth.x) break;
                int class_id = net.truth[t*(4 + 1) + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int class_index = entry_index(l, b, n, 4 + 1);
                        int obj_index = entry_index(l, b, n, 4);
                        float scale =  l.output[obj_index];
                        l.delta[obj_index] = l.noobject_scale * (0 - l.output[obj_index]);

                        float p = scale*get_hierarchy_probability(l.output + class_index, l.softmax_tree, class_id, l.w*l.h);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int class_index = entry_index(l, b, maxi, 4 + 1);
                    int obj_index = entry_index(l, b, maxi, 4);
                    delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.softmax_tree, l.class_scale, l.w*l.h, !l.softmax);

                    l.delta[obj_index] = 0;

                    l.skip_index[b] = 1;
                    break;
                }
            }
        }
        cuda_push_int_array(l.skip_index_gpu, l.skip_index, l.batch);
        cuda_push_array(l.delta_gpu, l.delta, l.outputs * l.batch);
    }
    delta_yolo_a_gpu(l.output_gpu, l.delta_gpu, l.biases_gpu, l.mask_gpu, net.truth_gpu, l.truths, l.skip_index_gpu, l.batch, l.w, l.h, l.n, l.max_boxes, l.classes, net.w, net.h, l.ignore_thresh, l.truth_thresh, l.map_gpu, l.class_scale, l.coord_scale, l.noobject_scale, l.object_scale, l.softmax, l.background, l.warmup, *(net.seen));
    delta_yolo_b_gpu(l.output_gpu, l.delta_gpu, l.biases_gpu, l.mask_gpu, net.truth_gpu, l.truths, l.skip_index_gpu, l.total, l.batch, l.w, l.h, l.n, l.max_boxes, l.classes, net.w, net.h, l.map_gpu, l.class_scale, l.coord_scale, l.object_scale, l.softmax, l.background, l.rescore, l.bias_match);
    cuda_pull_array(l.delta_gpu, l.delta, l.batch * l.outputs);
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
}

extern "C" void backward_yolo_layer_gpu(const layer l, network net) {
    if (l.logistic_derivative) {
        int n, b;
        for (b = 0; b < l.batch; ++b){
            for(n = 0; n < l.n; ++n){
                int index = b * l.outputs + n * l.w * l.h * (l.classes + 5) + 0 * l.w * l.h;
                gradient_array_gpu(l.output_gpu + index, 2*l.w*l.h, (activation_scheme){LOGISTIC}, l.delta_gpu);
            	index = b * l.outputs + n * l.w * l.h * (l.classes + 5) + (!l.background + 4) * l.w * l.h;
            	gradient_array_gpu(l.output_gpu + index, (l.background + l.classes)*l.w*l.h, (activation_scheme){LOGISTIC}, l.delta_gpu);
            }
        }
    }
    axpy_gpu(l.batch*l.inputs, l.impact, l.delta_gpu, 1, net.delta_gpu, 1);
}