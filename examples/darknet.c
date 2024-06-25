#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

extern void run_depth_compress(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_auto_colorizer(int argc, char **argv);
extern void run_super_resolution(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_regressor(int argc, char **argv);
extern void run_segmenter(int argc, char **argv);
extern void run_isegmenter(int argc, char **argv);
extern void run_lsd(int argc, char **argv);
extern void run_lp_superres(int argc, char **argv);

void convert_nonv_to_conv(char *cfgfile, char *weightfile, char* modules) {
    int i;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int nl = net->n;

    for (i = 0; i < nl; ++i) {
        if (net->layers[i].type == NORM_W_CONVOLUTIONAL) {
            net->layers[i].type = CONVOLUTIONAL;
            cuda_pull_array(net->layers[i].weights_gpu, net->layers[i].weights, net->layers[i].nweights);
        }
    }
    char new_file[256];
    sprintf(new_file, "%s_conv_cvt", weightfile);
    save_weights_upto(net, new_file, nl);
}

void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = 0;
    network *net = parse_network_cfg(cfgfile, 1);
    network *sum = parse_network_cfg(cfgfile, 1);

    char *weightfile = argv[4];   
    load_weights(sum, weightfile);

    int i, j;
    int n = argc - 5;
    for(i = 0; i < n; ++i){
        weightfile = argv[i+5];   
        load_weights(net, weightfile);
        for(j = 0; j < net->n; ++j){
            layer l = net->layers[j];
            layer out = sum->layers[j];
            if(l.type == CONVOLUTIONAL){
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if(l.batch_normalize){
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if(l.type == CONNECTED){
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n+1;
    for(j = 0; j < net->n; ++j){
        layer l = sum->layers[j];
        if(l.type == CONVOLUTIONAL){
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1./n, l.biases, 1);
            scal_cpu(num, 1./n, l.weights, 1);
                if(l.batch_normalize){
                    scal_cpu(l.n, 1./n, l.scales, 1);
                    scal_cpu(l.n, 1./n, l.rolling_mean, 1);
                    scal_cpu(l.n, 1./n, l.rolling_variance, 1);
                }
        }
        if(l.type == CONNECTED){
            scal_cpu(l.outputs, 1./n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1./n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

long numops(network *net)
{
    int i;
    long ops = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            ops += 2l * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w;
        } else if(l.type == CONNECTED){
            ops += 2l * l.inputs * l.outputs;
        } else if (l.type == RNN){
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        } else if (l.type == GRU){
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        } else if (l.type == LSTM){
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    return ops;
}

//nghiant: add param count for other layer types as well
long numparams(network *net) {
    int i;
    long params = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL) {
            params += l.c/l.groups * l.size * l.size * l.out_c + l.out_c;
        } else if (l.type == CONNECTED) {
            params += l.inputs * l.outputs;
        }
    }
    return params;
}

long num_gpu_buf(network* net) {
    int i;
    long gpu_buffer = 0;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        gpu_buffer += l.outputs;
    }
    return gpu_buffer;
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network *net = parse_network_cfg(cfgfile, 1);
    int i;
    double time=what_time_is_it_now();
    image im = make_image(net->w, net->h, net->c*net->batch);
    for(i = 0; i < tics; ++i){
        network_predict(net, im.data);
    }
    double t = what_time_is_it_now() - time;
    long ops = numops(net);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops/1000000000.);
    printf("FLOPS: %.2f Bn\n", (float)ops/1000000000.*tics/t);
    printf("Speed: %f sec/eval\n", t/tics);
    printf("Speed: %f Hz\n", tics/t);
}

void partial(char *cfgfile, char *weightfile, char* modules, char *outfile, int max)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 1, 1);
    save_weights_upto(net, outfile, max);
}

void modularize(char *cfgfile, char *weightfile, char* modules, int layer_name, int ignore_transform, int ignore_frozen)
{
    // gpu_index = 0;
    int i;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    if (net->pre_transformable) {
        net->pre_transform = 1;
        if (!ignore_transform) pre_transform_conv_params(net);
    }

    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (ignore_frozen && l.frozen == 1) continue;
        char modular_file_name[1024];
        if (weightfile) {
        	if (layer_name) sprintf(modular_file_name, "%s_%s", weightfile, l.name);
            else sprintf(modular_file_name, "%s_layer_%d", weightfile, i);
        } else {
        	if (layer_name) sprintf(modular_file_name, "init_%s", l.name);
            else sprintf(modular_file_name, "%s_init_layer_%d", basecfg(cfgfile), i);
        }
        if(l.type == CONVOLUTIONAL || l.type == DECONVOLUTIONAL){
            if (l.weight_transform.type && !ignore_transform) {
                swap_weight_transform(&l);
            }
            FILE* fp = fopen(modular_file_name, "wb");
            save_convolutional_weights(l, fp);
            fclose(fp);
        } if(l.type == CONNECTED){
            FILE* fp = fopen(modular_file_name, "wb");
            save_connected_weights(l, fp);
            fclose(fp);
        } if(l.type == BATCHNORM){
            FILE* fp = fopen(modular_file_name, "wb");
            save_batchnorm_weights(l, fp);
            fclose(fp);
        } if(l.type == RNN){
            FILE* fp = fopen(modular_file_name, "wb");
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
            fclose(fp);
        } if (l.type == LSTM) {
            FILE* fp = fopen(modular_file_name, "wb");
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.uo), fp);
            save_connected_weights(*(l.ug), fp);
            fclose(fp);
        } if (l.type == GRU) {
            FILE* fp = fopen(modular_file_name, "wb");
            save_connected_weights(*(l.wz), fp);
            save_connected_weights(*(l.wr), fp);
            save_connected_weights(*(l.wh), fp);
            save_connected_weights(*(l.uz), fp);
            save_connected_weights(*(l.ur), fp);
            save_connected_weights(*(l.uh), fp);
            fclose(fp);
        } if(l.type == CRNN){
            FILE* fp = fopen(modular_file_name, "wb");
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
            fclose(fp);
        }
    }
}

void slice_conv_weight(char *weightfile, int c, int groups, int size, int n, int slices, int batchnorm)
{
    gpu_index = 0;
    fprintf(stderr, "Slicing weights \033[0;32m%s\033[0m of size \033[0;32m%d x %d x %d (group: %d) x %d\033[0m into \033[0;32m%d\033[0m slices\n", weightfile, size, size, c, groups, n, slices);
    FILE* fp = fopen(weightfile, "rb");
    int num = size * size * c/groups;

    float* biases = calloc(n, sizeof(float));
    fread(biases, sizeof(float), n, fp);

    float* scales = 0;
    float* rolling_mean = 0;
    float* rolling_variance = 0;

    if (batchnorm) {
        scales = calloc(n, sizeof(float));
        fread(scales, sizeof(float), n, fp);

        rolling_mean = calloc(n, sizeof(float));
        fread(rolling_mean, sizeof(float), n, fp);

        rolling_variance = calloc(n, sizeof(float));
        fread(rolling_variance, sizeof(float), n, fp);
    }

    float* weights = calloc(n*num, sizeof(float));
    fread(weights, sizeof(float), n*num, fp);

    fclose(fp);

    assert(n % slices == 0);
    int len = n/slices;
    int i;

    char sid[256];
    for (i = 0; i < slices; ++i) {
        sprintf(sid, "%s_slices_%d_%d", weightfile, slices, i+1);
        fp = fopen(sid, "wb");

        fwrite(biases+i*len, sizeof(float), len, fp);
        if (batchnorm){
            fwrite(scales+i*len, sizeof(float), len, fp);
            fwrite(rolling_mean+i*len, sizeof(float), len, fp);
            fwrite(rolling_variance+i*len, sizeof(float), len, fp);
        }
        fwrite(weights+i*len*num, sizeof(float), len*num, fp);

        fprintf(stderr, "Saved slice: %s\n", sid);
        fclose(fp);
    }
    free(biases);
    free(scales);
    free(rolling_mean);
    free(rolling_variance);
    free(weights);
}

void merge_conv_weight(char *weightfile, char** list, int c, int groups, int size, int n, int slices, int batchnorm)
{
    gpu_index = 0;
    int i;

    assert(n % slices == 0);
    int len = n/slices;

    FILE* fp;

    int num = size * size * c / groups;

    float* biases = calloc(n, sizeof(float));
    float* scales = 0;
    float* rolling_mean = 0;
    float* rolling_variance = 0;

    if (batchnorm) {
        scales = calloc(n, sizeof(float));
        rolling_mean = calloc(n, sizeof(float));
        rolling_variance = calloc(n, sizeof(float));
    }

    float* weights = calloc(n*num, sizeof(float));

    for (i = 0; i < slices; ++i) {

	    fp = fopen(list[i], "rb");

	    fread(biases + i*len, sizeof(float), len, fp);

	    if (batchnorm) {
		    fread(scales + i*len, sizeof(float), len, fp);
		    fread(rolling_mean + i*len, sizeof(float), len, fp);
		    fread(rolling_variance + i*len, sizeof(float), len, fp);
	    }
	    fread(weights + num*i*len, sizeof(float), len*num, fp);

	    fclose(fp);
	}

    fp = fopen(weightfile, "wb");

    fwrite(biases, sizeof(float), n, fp);
    if (batchnorm){
        fwrite(scales, sizeof(float), n, fp);
        fwrite(rolling_mean, sizeof(float), n, fp);
        fwrite(rolling_variance, sizeof(float), n, fp);
    }
    fwrite(weights, sizeof(float), n*num, fp);

    fclose(fp);

    free(biases);
    if (batchnorm){
	    free(scales);
	    free(rolling_mean);
	    free(rolling_variance);
	}
    free(weights);
}

void form_weight_file(char *cfgfile, char *weightfile, char* modules, char *outfile)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    save_weights(net, outfile);
}

void sparsity(char *cfgfile, char *weightfile, char* modules, char* filename)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int i;
    
    char buff[256];
    char *input = buff;
    fprintf(stderr, "layer      weight sparsity   percentage     output sparsity    percentage\n");
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);

        float *X = sized.data;

        network_predict(net, X);
        int nl;
        for (nl = 0; nl < net->n; ++nl) {
            layer l = net->layers[nl];
            int s = l.out_c*l.out_h*l.out_w;

            float* out_cpu = malloc(s*sizeof(float));
            cuda_pull_array(l.output_gpu, out_cpu, s);

            int count = 0;
            for (i = 0; i < s; ++i) {
                if (out_cpu[i] == 0) ++count;
            }
            free(out_cpu);

            if (l.type == CONVOLUTIONAL) {
                int sw = l.nweights;

                float* w_cpu = malloc(sw*sizeof(float));
                if (l.weight_transform.type) {
                    cuda_pull_array(l.tran_weights_gpu, w_cpu, sw);
                } else {
                    cuda_pull_array(l.weights_gpu, w_cpu, sw);
                }

                int w_count = 0;
                for (i = 0; i < sw; ++i) {
                    if (w_cpu[i] == 0) ++w_count;
                }
                free(w_cpu);

                fprintf(stderr, "%5d    %8d/%8d     %6.2f      %8d/%8d       %6.2f\n", nl, w_count, sw, 100.0*(float)(w_count)/(float)(sw), count, s, 100.0*(float)(count)/(float)(s));
            } else {
                fprintf(stderr, "%5d        -   /    -            -       %8d/%8d       %6.2f\n", nl, count, s, 100.0*(float)(count)/(float)(s));
            }
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

void export_output(char *cfgfile, char *weightfile, char* modules, char* filename, int text)
{
    gpu_index = 0;

    char buff[256];
    char *input = buff;

    if(filename){
        strncpy(input, filename, 256);
    } else {
        fprintf(stderr, "No input found.\n");
        return;
    }

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int i;
    
    image im = load_image_color(input,0,0);

    //nghiant_remind: other loading methods for different tasks (detection, classification, ...) should be implemented
    image sized = letterbox_image(im, net->w, net->h);

    float *X = sized.data;

    char outf[256];
    FILE *file;
    float* out_cpu;
    int nl;

    //input
    if (text) {
        sprintf(outf,"%s_input.txt", weightfile);
        file = fopen(outf, "w");

        int count = 0;

        for (i = 0; i < net->w * net->h * net->c; ++i) {
            fprintf(file, "%.5f ", X[i]);
            ++count;
        }

    } else {
        sprintf(outf,"%s_input", weightfile);
        file = fopen(outf, "wb");
        fwrite(X, sizeof(float), net->w * net->h * net->c, file);
    }

    fprintf(stderr, "Exported as \033[0;32m%s\033[0m to %s (%d)\n", text ? "text" : "binary", outf, net->w * net->h * net->c);

    network_predict(net, X);

    for (nl = 0; nl < net->n; ++nl) {
    	fprintf(stderr, "Layer index: %d/%d\n", nl, net->n-1);
        layer l = net->layers[nl];

        out_cpu = malloc(l.outputs * sizeof(float));
        cuda_pull_array(l.output_gpu, out_cpu, l.outputs);

        if (text) {
        	sprintf(outf,"%s_output_%d.txt", weightfile, nl);
        	file = fopen(outf, "w");

	        int count = 0;

	        for (i = 0; i < l.outputs; ++i) {
	            fprintf(file, "%.5f ", out_cpu[i]);
	            ++count;
	        }


        } else {
        	sprintf(outf,"%s_output_%d", weightfile, nl);
        	file = fopen(outf, "wb");
        	fwrite(out_cpu, sizeof(float), l.outputs, file);
        }

        fprintf(stderr, "\033[1A");
        fprintf(stderr, "\033[K");
        

        fprintf(stderr, "Exported as \033[0;32m%s\033[0m to %s (%d)\n", text ? "text" : "binary", outf, l.outputs);

        free(out_cpu);
        fclose(file);
    }

    free_image(im);
    free_image(sized);
}

void print_weights(char *cfgfile, char *weightfile, char* modules, int n)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 1, 1);
    layer l = net->layers[n];
    int i, j;
    //printf("[");
    for(i = 0; i < l.n; ++i){
        //printf("[");
        for(j = 0; j < l.size*l.size*l.c; ++j){
            //if(j > 0) printf(",");
            printf("%g ", l.weights[i*l.size*l.size*l.c + j]);
        }
        printf("\n");
        //printf("]%s\n", (i == l.n-1)?"":",");
    }
    //printf("]");
}

void rgbgr_net(char *cfgfile, char *weightfile, char* modules, char *outfile)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char* modules, char *outfile)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize=1;
    l.scales = calloc(n, sizeof(float));
    for(j = 0; j < n; ++j){
        l.scales[j] = 1;
    }
    l.rolling_mean = calloc(n, sizeof(float));
    l.rolling_variance = calloc(n, sizeof(float));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char* modules, char *outfile)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int i;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == CONVOLUTIONAL && !l.batch_normalize){
            net->layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net->layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net->layers[i].batch_normalize=1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile, char* modules)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char* modules, char *outfile)
{
    gpu_index = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    int i;
    for (i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if ((l.type == DECONVOLUTIONAL || l.type == CONVOLUTIONAL) && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net->layers[i].batch_normalize=0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net->layers[i].batch_normalize=0;
        }
    }
    save_weights(net, outfile);
}

void mkimg(char *cfgfile, char *weightfile, char* modules, int h, int w, int num, char *prefix)
{
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    image *ims = get_weights(net->layers[0]);
    int n = net->layers[0].n;
    int z;
    for(z = 0; z < num; ++z){
        image im = make_image(h, w, 3);
        fill_image(im, .5);
        int i;
        for(i = 0; i < 100; ++i){
            image r = copy_image(ims[rand()%n]);
            rotate_image_cw(r, rand()%4);
            random_distort_image(r, 1, 1.5, 1.5);
            int dx = rand()%(w-r.w);
            int dy = rand()%(h-r.h);
            ghost_image(r, im, dx, dy);
            free_image(r);
        }
        char buff[256];
        sprintf(buff, "%s/gen_%d", prefix, z);
        save_image(im, buff);
        free_image(im);
    }
}

void visualize(char *cfgfile, char *weightfile, char* modules)
{
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    visualize_network(net);
}

void heatmap(char *cfgfile, char *weightfile, char* modules, char* filename, int layer_id, char* layer_name, int channel_id) {

    gpu_index = 0;

    char buff[256];
    char *input = buff;

    if(filename){
        strncpy(input, filename, 256);
    } else {
        fprintf(stderr, "No input found.\n");
        return;
    }

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    image im = load_image_color(input,0,0);

    //nghiant_remind: other loading methods for different tasks (detection, classification, ...) should be implemented
    image sized = letterbox_image(im, net->w, net->h);

    float *X = sized.data;
    float* out_cpu;
    int nl;

    network_predict(net, X);
    if (layer_name != 0) {
        for (nl = 0; nl < net->n; ++nl) {
            if (strcmp(layer_name, net->layers[nl].name) == 0) {
                layer_id = nl;
                break;
            }
        }
        if (nl == net->n) {
            fprintf(stderr, "Layer '%s' does not exist\n", layer_name);
            return;
        }
    }

    layer l = net->layers[layer_id];
    if (channel_id < 0 || channel_id >= l.out_c) {
        fprintf(stderr, "Channel ID %d out of range 0 ~ %d\n", channel_id, l.out_c-1);
        return;
    }

    fprintf(stderr, "Get output heatmap from channel %d of layer %d-%s (%s)\n", channel_id, layer_id, get_layer_string(l.type), l.name);

    out_cpu = malloc(l.outputs * sizeof(float));
    cuda_pull_array(l.output_gpu, out_cpu, l.outputs);

    image heat_im = real_to_heat_image(out_cpu + channel_id * l.out_w * l.out_h, l.out_w, l.out_h);

    save_image(heat_im, "heat_map");
    make_window("heat_map", 512, 512, 0);
    show_image(heat_im, "heat_map", 0);

    free(out_cpu);

    free_image(im);
    free_image(sized);
    free_image(heat_im);
}

int main(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }

    gpu_index = find_int_arg(argc, argv, "-i", 0);
    seed = find_int_arg(argc, argv, "-seed", time(0));
    srand(seed);
    fprintf(stderr, "GPU %2d\n", gpu_index);
    fprintf(stderr, "Seed %d\n", seed);
    cuda_set_device(gpu_index);

    if (0 == strcmp(argv[1], "average")){
        average(argc, argv);
    } else if (0 == strcmp(argv[1], "lsd")){
        run_lsd(argc, argv);
    } else if (0 == strcmp(argv[1], "lps")){
        run_lp_superres(argc, argv);
    } else if (0 == strcmp(argv[1], "detector") || 0 == strcmp(argv[1], "det")){
        run_detector(argc, argv);
    } else if (0 == strcmp(argv[1], "depth_compress") || 0 == strcmp(argv[1], "dc")){
        run_depth_compress(argc, argv);
    } else if (0 == strcmp(argv[1], "colorizer") || 0 == strcmp(argv[1], "color")){
        run_auto_colorizer(argc, argv);
    } else if (0 == strcmp(argv[1], "superres") || 0 == strcmp(argv[1], "sres")){
        run_super_resolution(argc, argv);
    } else if (0 == strcmp(argv[1], "classifier") || 0 == strcmp(argv[1], "cls")){
        run_classifier(argc, argv);
    } else if (0 == strcmp(argv[1], "regressor")){
        run_regressor(argc, argv);
    } else if (0 == strcmp(argv[1], "isegmenter")){
        run_isegmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "segmenter")){
        run_segmenter(argc, argv);
    } else if (0 == strcmp(argv[1], "3d")){
        composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *out = find_char_arg(argc, argv, "-out", 0);
        rgbgr_net(cfg, weights, modules, out);

    } else if (0 == strcmp(argv[1], "reset")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *out = find_char_arg(argc, argv, "-out", 0);
        reset_normalize_net(cfg, weights, modules, out);

    } else if (0 == strcmp(argv[1], "denormalize")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *out = find_char_arg(argc, argv, "-out", 0);
        denormalize_net(cfg, weights, modules, out);

    } else if (0 == strcmp(argv[1], "statistics")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        statistics_net(cfg, weights, modules);

    } else if (0 == strcmp(argv[1], "normalize")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *out = find_char_arg(argc, argv, "-out", 0);
        normalize_net(cfg, weights, modules, out);

    } else if (0 == strcmp(argv[1], "speed")){
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        int tic = find_int_arg(argc, argv, "-tic", 0);
        speed(cfg, tic);

    } else if (0 == strcmp(argv[1], "print")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        int layer_index = find_int_arg(argc, argv, "-index", 0);
        print_weights(cfg, weights, modules, layer_index);

    } else if (0 == strcmp(argv[1], "partial")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *out = find_char_arg(argc, argv, "-out", 0);
        int layer_index = find_int_arg(argc, argv, "-index", 0);
        partial(cfg, weights, modules, out, layer_index);

    } else if (0 == strcmp(argv[1], "average")){
        average(argc, argv);

    } else if (0 == strcmp(argv[1], "visualize")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        visualize(cfg, weights, modules);

    } else if (0 == strcmp(argv[1], "mkimg")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        mkimg(argv[2], argv[3], modules, atoi(argv[4]), atoi(argv[5]), atoi(argv[6]), argv[7]);

    } else if (0 == strcmp(argv[1], "modularize") || 0 == strcmp(argv[1], "mod")) {
    	int layer_name = find_arg(argc, argv, "-name");
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        int ignore_transform = find_arg(argc, argv, "-ignore_transform");
        int ignore_frozen = find_arg(argc, argv, "-ignore_frozen");
    	modularize(cfg, weights, modules, layer_name, ignore_transform, ignore_frozen);

    } else if (0 == strcmp(argv[1], "convert_nonv_to_conv") || 0 == strcmp(argv[1], "cntc")) {
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        convert_nonv_to_conv(cfg, weights, modules);

    } else if (0 == strcmp(argv[1], "slice")) {
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        int c = find_int_arg(argc, argv, "-in_c", 16);
        int groups = find_int_arg(argc, argv, "-groups", 1);
        int size = find_int_arg(argc, argv, "-size", 3);
        int n = find_int_arg(argc, argv, "-out_c", 16);
        int slices = find_int_arg(argc, argv, "-slices", 4);
        int no_batchnorm = find_arg(argc, argv, "-no_batchnorm");
        slice_conv_weight(weights, c, groups, size, n, slices, !no_batchnorm);

    } else if (0 == strcmp(argv[1], "merge")) {
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        int slices = find_int_arg(argc, argv, "-slices", 4);
        char **list = calloc(slices, sizeof(char*));
        int i = 0;
        for (i = 0; i < slices; ++i) {
	        list[i] = find_char_2arg(argc, argv, "-l", "--list", 0);
	        if (list[i] == 0) {
	        	fprintf(stderr, "Not enough input files\n");
	        	return 0;
	        }
	        fprintf(stderr, "%s\n", list[i]);
	    }
        int c = find_int_arg(argc, argv, "-in_c", 16);
        int groups = find_int_arg(argc, argv, "-groups", 1);
        int size = find_int_arg(argc, argv, "-size", 3);
        int n = find_int_arg(argc, argv, "-out_c", 16);
        int no_batchnorm = find_arg(argc, argv, "-no_batchnorm");
        merge_conv_weight(weights, list, c, groups, size, n, slices, !no_batchnorm);
        free(list);

    } else if (0 == strcmp(argv[1], "form")){
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *out = find_char_arg(argc, argv, "-out", 0);
        form_weight_file(cfg, weights, modules, out);

    } else if (0 == strcmp(argv[1], "sparsity")) {
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
    	sparsity(cfg, weights, modules, filename);

    } else if (0 == strcmp(argv[1], "export_output")) {
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
    	int text = find_arg(argc, argv, "-text");
    	export_output(cfg, weights, modules, filename, text);

    } else if (0 == strcmp(argv[1], "heatmap")) {
        char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);

        int layer_id = find_int_arg(argc, argv, "-li", 0);
        char *layer_name = find_char_2arg(argc, argv, "-ln", "--layer_name", 0);
        int channel_id = find_int_arg(argc, argv, "-ci", 0);
        
        heatmap(cfg, weights, modules, filename, layer_id, layer_name, channel_id);

    } else if (0 == strcmp(argv[1], "parse")){
        char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
        fprintf(stderr, "%s\n", basecfg(cfg));
        network* net = parse_network_cfg(cfg, 1);

        fprintf(stderr, "\n====Complexity====\n");
        long params = numparams(net);
        long ops = numops(net);
        long gpu_buffer = num_gpu_buf(net);

        fprintf(stderr, "Bil. Oper   \033[0;32m%6.2f\033[0m\n", (float)ops/1000000000.);
        fprintf(stderr, "Mil. Param  \033[0;32m%6.2f\033[0m\n", (float)params/1000000.);
        fprintf(stderr, "GPU Buffer  \033[0;32m%6.2f\033[0m\n", (float)gpu_buffer/250000000.);

        load_args args = get_base_args(net);
        summarize_data_augmentation_options(args);

    } else if (0 == strcmp(argv[1], "seen")){
        char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
        FILE *fp = fopen(weights, "rb");
        if(fp) {
            int major;
            int minor;
            int revision;
            size_t seen;

            fread(&major, sizeof(int), 1, fp);
            fread(&minor, sizeof(int), 1, fp);
            fread(&revision, sizeof(int), 1, fp);
            if ((major*10 + minor) >= 2 && major < 1000 && minor < 1000){
                fread(&seen, sizeof(size_t), 1, fp);
            } else {
                int iseen = 0;
                fread(&iseen, sizeof(int), 1, fp);
                seen = iseen;
            }
            fprintf(stderr, "Passed: \033[0;32m%16zu\033[0m iters with batch =  1\n", seen);
            fprintf(stderr, "Passed: \033[0;32m%16zu\033[0m iters with batch = 64\n", seen/64);
        }
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);

    }
    return 0;
}