#include "darknet.h"
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include "image.h"
#include "blas.h"

void train_lp_superres(char* datacfg, char *cfgfile, char *weightfile, char* modules, int *gpus, int ngpus, int clear, int info)
{
    list *options = read_data_cfg(datacfg);
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *info_directory = option_find_str(options, "info", "/info/");
    int us_factor = option_find_int(options, "us_factor", 2);
    char *base = basecfg(cfgfile);

    char infofile_name[256];
    sprintf(infofile_name, "%s/%s_%u.info", info_directory, base, (unsigned)time(NULL));

    fprintf(stderr, "%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network*));

    int i;
    for(i = 0; i < ngpus; ++i){
        cuda_set_device(gpus[i]);
        nets[i] = load_network(cfgfile, weightfile, modules, clear, 0);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    list *plist = get_paths(train_list);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    double time;

    load_args args = get_base_args(net);

    args.us_factor = us_factor;
    args.threads = 32;
    args.paths = paths;
    args.n = imgs;
    args.m = N;
    args.type = LP_SUPERRES_DATA;

    data train, buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    summarize_data_augmentation_options(args);

    FILE* infofile = 0;
    if (info) {
        infofile = fopen(infofile_name, "wb");
        // save_training_info(infofile, net, 1, N);
        save_training_info(infofile, net, 1, 599264);
    }
    fprintf(stderr, "Learning Rate: %g, Decay: %g\n", net->learning_rate, net->decay);

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        time = what_time_is_it_now();
        float loss = 0;
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%7ld: ", get_current_batch(net));
        int il;
        for (il = 0; il < net->n_loss; ++il) {
            printf("%9.6f (%d-%s)  ", net->sub_loss[il], net->sub_loss_id[il], get_layer_string(net->layers[net->sub_loss_id[il]].type));
        }
        // printf("%9.6f (avg)  %8.6f (lr)  %5.2f (s)  %10ld (imgs)  %6.2f (epochs)\n", avg_loss, get_current_rate(net), what_time_is_it_now()-time, get_current_batch(net)*imgs, (float)(*net->seen)/N);
        printf("%9.6f (avg)  %8.6f (lr)  %5.2f (s)  %10ld (imgs)  %6.2f (epochs)\n", avg_loss, get_current_rate(net), what_time_is_it_now()-time, get_current_batch(net)*imgs, (float)(*net->seen)/599264);
        
        // if (info) save_training_info(infofile, net, 0, N);
        if (info) save_training_info(infofile, net, 0, 599264);

        free_data(train);

        if (get_current_batch(net) % 1000 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
            if (info) {
                fclose(infofile);
                infofile = fopen(infofile_name, "ab");
                fprintf(stderr, "Save training info to %s\n", infofile_name);
            }
        }

        if (get_current_batch(net) % 10000 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s_%lu.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);
    if (info) {
        fclose(infofile);
        fprintf(stderr, "Training info: %s\n", infofile_name);
    }

    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void train_lp_superres_tf(char* datacfg, char *cfgfile, char *weightfile, char* modules, int *gpus, int ngpus, int clear, int info)
{
    list *options = read_data_cfg(datacfg);
    char *train_directory = option_find_str(options, "tf_train", "data/");
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *im_prefix = option_find_str(options, "tf_train_im_prefix", "im");
    char *gt_prefix = option_find_str(options, "tf_train_gt_prefix", "gt");
    int N = option_find_int(options, "tf_train_num", 0);
    assert(N > 0);

    char *info_directory = option_find_str(options, "info", "/info/");
    int us_factor = option_find_int(options, "us_factor", 2);
    char *base = basecfg(cfgfile);

    char infofile_name[256];
    sprintf(infofile_name, "%s/%s_%u.info", info_directory, base, (unsigned)time(NULL));

    fprintf(stderr, "%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network*));

    int i;
    for(i = 0; i < ngpus; ++i){
        cuda_set_device(gpus[i]);
        nets[i] = load_network(cfgfile, weightfile, modules, clear, 0);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];

    int netw = net->w;
    int neth = net->h;

    int gtw = us_factor * netw;
    int gth = us_factor * neth;

    float** all_data_im = (float**)calloc(N, sizeof(float*));
    float** all_data_gt = (float**)calloc(N, sizeof(float*));
    fprintf(stderr, "\n");
    for (i = 0; i < N; ++i) {
        all_data_im[i] = (float*)calloc(netw * neth, sizeof(float));
        all_data_gt[i] = (float*)calloc(gtw  * gth , sizeof(float));
        
        char file_name[256];
        FILE *fid;

        sprintf(file_name, "%s%s_%d", train_directory, im_prefix, i);
        fid = fopen(file_name, "rb");
        fread((void*)(all_data_im[i]), sizeof(float), netw*neth, fid);
        fclose(fid);
        sprintf(file_name, "%s%s_%d", train_directory, gt_prefix, i);
        fid = fopen(file_name, "rb");
        fread((void*)(all_data_gt[i]), sizeof(float), gtw * gth, fid);
        fclose(fid);

        fprintf(stderr, "\033[1A");
        fprintf(stderr, "\033[K");
    	fprintf(stderr, "[ INFO ] data preloading (memory intensive) : %6d/%6d\n", i+1, N);
    }
    
    int imgs = net->batch * net->subdivisions * ngpus;
    // assert(N % imgs == 0);

    double time;

    load_args args = get_base_args(net);

    args.us_factor = us_factor;
    args.threads = 32;
    args.all_data_im = all_data_im;
    args.all_data_gt = all_data_gt;
    args.n = imgs;
    args.m = N;
    args.type = LP_SUPERRES_TF_DATA;

    data train, buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    summarize_data_augmentation_options(args);

    FILE* infofile = 0;
    if (info) {
        infofile = fopen(infofile_name, "w");
        save_training_info(infofile, net, 1, N);
    }
    fprintf(stderr, "Learning Rate: %g, Decay: %g\n", net->learning_rate, net->decay);

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        time = what_time_is_it_now();
        float loss = 0;
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%7ld: ", get_current_batch(net));
        int il;
        for (il = 0; il < net->n_loss; ++il) {
            printf("%9.6f (%d-%s)  ", net->sub_loss[il], net->sub_loss_id[il], get_layer_string(net->layers[net->sub_loss_id[il]].type));
        }
        printf("%9.6f (avg)  %8.6f (lr)  %5.2f (s)  %10ld (imgs)  %6.2f (epochs)\n", avg_loss, get_current_rate(net), what_time_is_it_now()-time, get_current_batch(net)*imgs, (float)(*net->seen)/N);
        
        if (info) save_training_info(infofile, net, 0, N);

        // free_data(train);

        if (get_current_batch(net) % 1000 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
            if (info) {
                fclose(infofile);
                infofile = fopen(infofile_name, "ab");
                fprintf(stderr, "Save training info to %s\n", infofile_name);
            }
        }

        if (get_current_batch(net) % 100 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s_%lu.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }
    }

    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);
    if (info) {
        fclose(infofile);
        fprintf(stderr, "Training info: %s\n", infofile_name);
    }

    free_network(net);
    free(base);
    free(all_data_im);
    free(all_data_gt);
}

void train_lp_superres_tf_less(char* datacfg, char *cfgfile, char *weightfile, char* modules, int *gpus, int ngpus, int clear, int info)
{
    list *options = read_data_cfg(datacfg);
    char *train_directory = option_find_str(options, "tf_train", "data/");
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *im_prefix = option_find_str(options, "tf_train_im_prefix", "im");
    char *gt_prefix = option_find_str(options, "tf_train_gt_prefix", "gt");
    int N = option_find_int(options, "tf_train_num", 0);
    assert(N > 0);

    char *info_directory = option_find_str(options, "info", "/info/");
    int us_factor = option_find_int(options, "us_factor", 2);
    char *base = basecfg(cfgfile);

    char infofile_name[256];
    sprintf(infofile_name, "%s/%s_%u.info", info_directory, base, (unsigned)time(NULL));

    fprintf(stderr, "%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network*));

    int i;
    for(i = 0; i < ngpus; ++i){
        cuda_set_device(gpus[i]);
        nets[i] = load_network(cfgfile, weightfile, modules, clear, 0);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    double time;

    load_args args = get_base_args(net);

    args.us_factor = us_factor;
    args.threads = 32;
    args.train_directory = train_directory;
    args.im_prefix = im_prefix;
    args.gt_prefix = gt_prefix;
    args.n = imgs;
    args.m = N;
    args.type = LP_SUPERRES_TF_LESS_DATA;

    data train, buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    summarize_data_augmentation_options(args);

    FILE* infofile = 0;
    if (info) {
        infofile = fopen(infofile_name, "w");
        save_training_info(infofile, net, 1, N);
    }
    fprintf(stderr, "Learning Rate: %g, Decay: %g\n", net->learning_rate, net->decay);

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        time = what_time_is_it_now();
        float loss = 0;
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }

        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%7ld: ", get_current_batch(net));
        int il;
        for (il = 0; il < net->n_loss; ++il) {
            printf("%9.6f (%d-%s)  ", net->sub_loss[il], net->sub_loss_id[il], get_layer_string(net->layers[net->sub_loss_id[il]].type));
        }
        printf("%9.6f (avg)  %8.6f (lr)  %5.2f (s)  %10ld (imgs)  %6.2f (epochs)\n", avg_loss, get_current_rate(net), what_time_is_it_now()-time, get_current_batch(net)*imgs, (float)(*net->seen)/N);
        
        if (info) save_training_info(infofile, net, 0, N);

        free_data(train);

        if (get_current_batch(net) % 1000 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
            if (info) {
                fclose(infofile);
                infofile = fopen(infofile_name, "ab");
                fprintf(stderr, "Save training info to %s\n", infofile_name);
            }
        }

        if (get_current_batch(net) % 5000 == 0) {
            char buff[256];
            sprintf(buff, "%s/%s_%lu.weights", backup_directory, base, get_current_batch(net));
            save_weights(net, buff);
        }
    }

    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);
    if (info) {
        fclose(infofile);
        fprintf(stderr, "Training info: %s\n", infofile_name);
    }

    free_network(net);
    free(base);
}

void valid_lp_superres(char* datacfg, char *cfgfile, char *weightfile, char* modules) {
    list *options = read_data_cfg(datacfg);
    char *val_list = option_find_str(options, "valid", "data/valid.list");
    list *plist = get_paths(val_list);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    fprintf(stderr, "PSNR\n");
    int i,j,k;
    for (k = 0; k < N; ++k) {
    	image img = load_image_color(paths[k], 0, 0);
    	image gt = copy_image(img);
    	resize_network(net, img.w/2, img.h/2);
    	image resized = resize_image(img, net->w, net->h);

	    rgb_to_ycbcr(resized);
	    rgb_to_ycbcr(img);
	    rgb_to_ycbcr(gt);
	    float* X = resized.data;
	    float* Y = network_predict(net, X);
	    memcpy(img.data, Y, 4*net->w * net->h * sizeof(float));
	    int M = (img.w-4) * (img.h-4);
	    float sum = 0;
	    for (j = 2; j < img.h-2; ++j) {
            for (i = 2; i < img.w-2; ++i) {
                sum += (img.data[j*img.w+i] - gt.data[j*img.w+i])*(img.data[j*img.w+i] - gt.data[j*img.w+i]);
            }
	    }
	    fprintf(stderr, "%f\n", 20*log10(1/sqrt(sum/M)));

    }
}


void valid_tf_lp_superres(char* datacfg, char *cfgfile, char *weightfile, char* modules) {
    list *options = read_data_cfg(datacfg);

    char *val_list = option_find_str(options, "tf_valid_im", "data/tf_im.list");
    list *plist = get_paths(val_list);

    char *gt_list = option_find_str(options, "tf_valid_gt", "data/tf_gt.list");
    list *gt_plist = get_paths(gt_list);

    int N = plist->size;
    char **paths = (char **)list_to_array(plist);
    char **gt_paths = (char **)list_to_array(gt_plist);

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    fprintf(stderr, "PSNR\n");
    int i,j,k;
    float total_psnr = 0;
    
    int nl = net->n;
    float* max = (float*)calloc(nl, sizeof(float));
    float* min = (float*)calloc(nl, sizeof(float));
    for (i = 0; i < nl; ++i) {
        max[i] = -9999;
        min[i] = 9999;
    }
    float* maxw = (float*)calloc(nl, sizeof(float));
    float* minw = (float*)calloc(nl, sizeof(float));
    for (i = 0; i < nl; ++i) {
        maxw[i] = -9999;
        minw[i] = 9999;
    }

    for (k = 0; k < N; ++k) {
        float fnw, fnh;
        int nw, nh;
        FILE* tf_input = fopen(paths[k], "rb");
        fread(&fnw, sizeof(float), 1, tf_input);
        fread(&fnh, sizeof(float), 1, tf_input);
        nw = (int)fnw;
        nh = (int)fnh;
        float *X = (float*)calloc(nw*nh, sizeof(float));
        fread(X, 4, nw*nh, tf_input);
        fclose(tf_input);

        float fimw, fimh;
        int imw, imh;
        FILE* tf_gt = fopen(gt_paths[k], "rb");
        fread(&fimw, sizeof(float), 1, tf_gt);
        fread(&fimh, sizeof(float), 1, tf_gt);
        imw = (int)fimw;
        imh = (int)fimh;
        float *G = (float*)calloc(imw*imh, sizeof(float));
        fread(G, 4, imw*imh, tf_gt);
        fclose(tf_gt);

        resize_network(net, nw, nh);
        float* Y = network_predict(net, X);

        for (j = 0; j < nl; ++j) {
            layer l = net->layers[j];
            cuda_pull_array(l.output_gpu, l.output, l.outputs);
            int z;
            for (z = 0; z < l.outputs; ++z) {
                float zz = l.output[z];
                max[j] = zz > max[j] ? zz : max[j];
                min[j] = zz < min[j] ? zz : min[j];
            }
            // fprintf(stderr, "%.3f %.3f\n", min[j], max[j]);
        }

        for (j = 0; j < nl; ++j) {
            layer l = net->layers[j];
            if (l.type != CONVOLUTIONAL) continue;
            cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
            int z;
            for (z = 0; z < l.nweights; ++z) {
                float zz = l.weights[z];
                maxw[j] = zz > maxw[j] ? zz : maxw[j];
                minw[j] = zz < minw[j] ? zz : minw[j];
            }
            // fprintf(stderr, "%.3f %.3f\n", minw[j], maxw[j]);
        }

        int M = (imw-4) * (imh-4);
        float sum = 0;
        for (j = 2; j < imh-2; ++j) {
            for (i = 2; i < imw-2; ++i) {
                sum += (Y[j*imw+i] - G[j*imw+i]) * (Y[j*imw+i] - G[j*imw+i]);
            }
        }
        sum = 20*log10(1/sqrt(sum/M));
        fprintf(stderr, "%f\n", sum);
        total_psnr += sum;

        free(X);
        free(G);
    }

    for (j = 0; j < nl; ++j) {
        layer l = net->layers[j];
        if (l.type != CONVOLUTIONAL) continue;
        fprintf(stderr, "%.3f %.3f\n", min[j], max[j]);
    }

    for (j = 0; j < nl; ++j) {
        layer l = net->layers[j];
        if (l.type != CONVOLUTIONAL) continue;
        fprintf(stderr, "%.3f %.3f\n", minw[j], maxw[j]);
    }

    printf("Mean: %f\n", total_psnr/N);
}



void custom_lp(char* datacfg, char *cfgfile, char *weightfile, char* modules) {
    list *options = read_data_cfg(datacfg);

    char *val_list = option_find_str(options, "tf_valid_im", "data/tf_im.list");
    list *plist = get_paths(val_list);

    char *gt_list = option_find_str(options, "tf_valid_gt", "data/tf_gt.list");
    list *gt_plist = get_paths(gt_list);

    int N = plist->size;
    char **paths = (char **)list_to_array(plist);
    char **gt_paths = (char **)list_to_array(gt_plist);

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    // int v;

    fprintf(stderr, "PSNR\n");
    int i,k;
    float total_psnr = 0;
    
    // int nl = 10;
    // float* max = (float*)calloc(nl, sizeof(float));
    // float* min = (float*)calloc(nl, sizeof(float));
    // for (i = 0; i < nl; ++i) {
    //     max[i] = -9999;
    //     min[i] = 9999;
    // }

    for (k = 0; k < 1; ++k) {
        int nw, nh;
        FILE* tf_input = fopen(paths[k], "r");
        fscanf(tf_input, "%d", &nw);
        fscanf(tf_input, "%d", &nh);
        float *X = (float*)calloc(nw*nh, sizeof(float));
        for (i = 0; i < nw * nh; ++i) {
            fscanf(tf_input, "%f", X+i);
        }
        fclose(tf_input);

        int imw, imh;
        FILE* tf_gt = fopen(gt_paths[k], "r");
        fscanf(tf_gt, "%d", &imw);
        fscanf(tf_gt, "%d", &imh);
        float *G = (float*)calloc(imw*imh, sizeof(float));
        for (i = 0; i < imw * imh; ++i) {
            fscanf(tf_gt, "%f", G+i);
        }
        fclose(tf_gt);

        resize_network(net, nw, nh);
        float* Y = network_predict(net, X);
        for (i = 0; i < net->layers[net->n-1].outputs; ++i) {
            fprintf(stderr, "%f\n", Y[i]);
        }

        // int M = (imw-4) * (imh-4);
        // float sum = 0;
        // for (j = 2; j < imh-2; ++j) {
        //     for (i = 2; i < imw-2; ++i) {
        //         sum += (Y[j*imw+i] - G[j*imw+i]) * (Y[j*imw+i] - G[j*imw+i]);
        //     }
        // }
        // sum = 20*log10(1/sqrt(sum/M));
        // fprintf(stderr, "%f\n", sum);
        // total_psnr += sum;

        free(X);
        free(G);
    }

    // int nnn = 18;

    // fprintf(stderr, "v/norm_v \n");
    // for (v = 0; v < nnn; ++v) {
    //     fprintf(stderr, "Layer: %2d ~ ", v);
    //     layer l = net->layers[v];
    //     float max = -9999;
    //     float min = 9999;
    //     norm_w_normalize_gpu(l.weights_v_gpu, l.weights_v_norm_gpu, l.nweights_v/l.n, l.n);
    //     norm_w_norm_div_weights_gpu(l.weights_gpu, l.weights_v_gpu, l.weights_v_norm_gpu, l.nweights_v/l.n, l.n);
    //     cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    //     int z;
    //     for (z = 0; z < l.nweights; ++z) {
    //         float zz = l.weights[z];
    //         max = zz > max ? zz : max;
    //         min = zz < min ? zz : min;
    //     }
    //     fprintf(stderr, "%f\t%f\n", min, max);
    // }

    // fprintf(stderr, "weight_g \n");
    // for (v = 0; v < nnn; ++v) {
    //     fprintf(stderr, "Layer: %2d ~ ", v);
    //     layer l = net->layers[v];
    //     float max = -9999;
    //     float min = 9999;
    //     cuda_pull_array(l.weights_g_gpu, l.weights_g, l.nweights_g);
    //     int z;
    //     for (z = 0; z < l.nweights_g; ++z) {
    //         float zz = l.weights_g[z];
    //         max = zz > max ? zz : max;
    //         min = zz < min ? zz : min;
    //     }
    //     fprintf(stderr, "%f\t%f\n", min, max);
    // }

    // fprintf(stderr, "act\n");
    // for (j = 0; j < nl; ++j) {
    //     fprintf(stderr, "Layer: %2d ~ %f\t%f\n", j, min[j], max[j]);
    // }

    printf("Mean: %f\n", total_psnr/N);
}


void valid_trainset_tf_lp_superres(char* datacfg, char *cfgfile, char *weightfile, char* modules) {
    list *options = read_data_cfg(datacfg);

    char *val_list = option_find_str(options, "tf_valid_im", "data/tf_im.list");
    list *plist = get_paths(val_list);

    char *gt_list = option_find_str(options, "tf_valid_gt", "data/tf_gt.list");
    list *gt_plist = get_paths(gt_list);

    int N = plist->size;
    char **paths = (char **)list_to_array(plist);
    char **gt_paths = (char **)list_to_array(gt_plist);

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    fprintf(stderr, "PSNR\n");
    int i,j,k;
    int nw = 21;
    int nh = 21;
    int imw = 42;
    int imh = 42;
    
    float *X = (float*)calloc(nw*nh, sizeof(float));
    float *G = (float*)calloc(imw*imh, sizeof(float));

    fprintf(stderr, "\n");
    for (k = 0; k < N; ++k) {
        FILE* tf_input = fopen(paths[k], "rb");
        fread((void*)(X), sizeof(float), nw*nw, tf_input);
        fclose(tf_input);

        FILE* tf_gt = fopen(gt_paths[k], "rb");
        fread((void*)(G), sizeof(float), imw*imh, tf_gt);
        fclose(tf_gt);

        resize_network(net, nw, nh);
        float* Y = network_predict(net, X);

        int M = (imw-4) * (imh-4);
        float sum = 0;
        for (j = 2; j < imh-2; ++j) {
            for (i = 2; i < imw-2; ++i) {
                sum += (Y[j*imw+i] - G[j*imw+i]) * (Y[j*imw+i] - G[j*imw+i]);
            }
        }
        // sum = 20*log10(1/sqrt(sum/M));
        sum = sum / M;
        fprintf(stderr, "\033[1A");
        fprintf(stderr, "\033[K");
    	fprintf(stderr, "[ INFO ] data checking: %6d/%6d\n", k, N);
        if (sum > 1) fprintf(stderr, "%d - %f\n\n", k, sum);
    }
}

void self_test_lp_superres(char *cfgfile, char *weightfile, char* modules, char* img_name)
{
	image img = load_image_color(img_name, 0, 0);
	// image img = transpose_image(tim);
	image gt_img = copy_image(img);
    save_image(img, "gt");
    char *base = basecfg(cfgfile);

    fprintf(stderr, "%s\n", base);
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    resize_network(net, img.w/2, img.h/2);
	
	image resized = resize_image(img, net->w, net->h);
    rgb_to_ycbcr(resized);
    rgb_to_ycbcr(img);
    rgb_to_ycbcr(gt_img);

	float* X = resized.data;
	float* Y = network_predict(net, X);
	memcpy(img.data, Y, 4*net->w * net->h * sizeof(float));

    int i,j;
    int M = (img.w-4) * (img.h-4);
    float sum = 0;
    for (j = 2; j < img.h-2; ++j) {
        for (i = 2; i < img.w-2; ++i) {
            sum += (img.data[j*img.w+i] - gt_img.data[j*img.w+i])*(img.data[j*img.w+i] - gt_img.data[j*img.w+i]);
        }
    }
    fprintf(stderr, "PSNR: %f\n", 20*log10(1/sqrt(sum/M)));

    ycbcr_to_rgb(img);
    make_window("lpsr", 2*net->w, 2*net->h, 0);
    show_image(img, "lpsr", 0);
    save_image(img, "lpsr");

    free_image(resized);
    free_image(img);
    free_image(gt_img);

}

void test_lp_superres(char *cfgfile, char *weightfile, char* modules, char* img_name)
{
	image img = load_image_color(img_name, 0, 0);

    char *base = basecfg(cfgfile);

    fprintf(stderr, "%s\n", base);
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
	
	image resized = resize_image(img, net->w, net->h);
	image res2x_gt = resize_image(resized, net->w*2, net->h*2);
	image res2x = resize_image(resized, net->w*2, net->h*2);
    
    rgb_to_ycbcr(resized);
    rgb_to_ycbcr(res2x);

	float* X = resized.data;
	float* Y = network_predict(net, X);
	memcpy(res2x.data, Y, 4*net->w * net->h * sizeof(float));
	ycbcr_to_rgb(res2x);

	make_window("lpsr", net->w*2, net->h*2, 0);
    show_image(res2x, "lpsr", 0);
    save_image(res2x, "lpsr");
    save_image(res2x_gt, "gt");

    free_image(resized);
    free_image(res2x);
    free_image(img);

}

void test_open_file() {
    FILE* fid = fopen("gt_0", "r");
    float* a = (float*)calloc(42*42, sizeof(float));
    int i;
    fread((void*)(a), sizeof(float), 42*42, fid);
    for (i = 0; i < 42*42; ++i) {
        fprintf(stderr, "%f\n", a[i]);
    }
}

void custom_lps() {
	char* cfgfile = "cfg/svdsr10_2x_4bw_mean.cfg";
	char* weights = "backup/svdsr10_2x_4bw_mean.backup";

    network *net = load_network(cfgfile, weights, 0, 0, 1);

    int i;

    layer l = net->layers[0];
    if (net->pre_transformable) {
        net->pre_transform = 1;
        pre_transform_conv_params(net);
    }
    swap_weight_transform(&l);
    fprintf(stderr, "%d\n", l.n_coeff);
    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    for (i = 0; i < 4; ++i) {
        fprintf(stderr, "%f\n", l.weights[i]);
    }
}

void run_lp_superres(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s %s [train/test/valid]    --data [data_cfg] --config [cfg] --weight [weight (optional)] --module [module (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    char *data = find_char_2arg(argc, argv, "-d", "--data", 0);
    char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
    char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
    char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
    
    if (0==strcmp(argv[2], "train")) {
	    int info = find_arg(argc, argv, "-info");
	    int clear = find_arg(argc, argv, "-clear");
        train_lp_superres(data, cfg, weights, modules, gpus, ngpus, clear, info);
    } else if (0==strcmp(argv[2], "ttf")) {
        int info = find_arg(argc, argv, "-info");
        int clear = find_arg(argc, argv, "-clear");
        train_lp_superres_tf(data, cfg, weights, modules, gpus, ngpus, clear, info);
    } else if (0==strcmp(argv[2], "ttfless")) {
        int info = find_arg(argc, argv, "-info");
        int clear = find_arg(argc, argv, "-clear");
        train_lp_superres_tf_less(data, cfg, weights, modules, gpus, ngpus, clear, info);
    } else if (0==strcmp(argv[2], "test")) {
    	char *img_name = find_char_2arg(argc, argv, "-im", "--image", 0);
        test_lp_superres(cfg, weights, modules, img_name);
    } else if (0==strcmp(argv[2], "self")) {
    	char *img_name = find_char_2arg(argc, argv, "-im", "--image", 0);
        self_test_lp_superres(cfg, weights, modules, img_name);
    } else if (0==strcmp(argv[2], "valid")) {
        valid_lp_superres(data, cfg, weights, modules);
    } else if (0==strcmp(argv[2], "vtf")) {
        valid_tf_lp_superres(data, cfg, weights, modules);
    } else if (0==strcmp(argv[2], "vttf")) {
        valid_trainset_tf_lp_superres(data, cfg, weights, modules);
    } else if (0==strcmp(argv[2], "custom")) {
        custom_lps();
    } else if (0==strcmp(argv[2], "tof")) {
        test_open_file();
    } else if (0==strcmp(argv[2], "custom_lp")) {
        custom_lp(data, cfg, weights, modules);
    } else {
        fprintf(stderr, "none binary found\n");
    }
}
