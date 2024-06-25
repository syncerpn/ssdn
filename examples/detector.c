#include "darknet.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "blas.h"
#include "convolutional_layer.h"
#include "quantization_layer.h"

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void train_detector(char *datacfg, char *cfgfile, char *weightfile, char* modules, int *gpus, int ngpus, int clear, int info)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *info_directory = option_find_str(options, "info", "/info/");

    char* label_dir = option_find_str_quiet(options, "label_dir", "labels");

    char *base = basecfg(cfgfile);
    
    char infofile_name[256];
    sprintf(infofile_name, "%s/%s_%u.info", info_directory, base, (unsigned)time(NULL));
    
    printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network));

    int i;
    for(i = 0; i < ngpus; ++i){
        cuda_set_device(gpus[i]);
        nets[i] = load_network(cfgfile, weightfile, modules, clear, 0);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;
    data train, buffer;

    layer l = net->layers[net->n - 1];

    int classes = l.classes;

    list *plist = get_paths(train_images);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);

    args.coords = l.coords;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.num_boxes = l.max_boxes;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;

    args.label_dir = label_dir;
    
    summarize_data_augmentation_options(args);

    pthread_t load_thread = load_data(args);
    double time;
    int count = 0;

    FILE* infofile = 0;
    if (info) {
        infofile = fopen(infofile_name, "wb");
        save_training_info(infofile, net, 1, N);
    }
    printf("Learning Rate: %g, Decay: %g\n", net->learning_rate, net->decay);

    while(get_current_batch(net) < net->max_batches){
        if(net->random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32;
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        time=what_time_is_it_now();
        float loss = 0;

        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }

        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);

        printf("%7ld: ", get_current_batch(net));
        int il;
        for (il = 0; il < net->n_loss; ++il) {
            printf("%9.6f (%d-%s)  ", net->sub_loss[il], net->sub_loss_id[il], get_layer_string(net->layers[net->sub_loss_id[il]].type));
        }
        printf("%9.6f (avg)  %8.6f (lr)  %5.2f (s)  %10d (imgs)  %6.2f (epochs)\n", avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs, (float)(*net->seen)/N);

        if (info) save_training_info(infofile, net, 0, N);

        if (i%100==0) {

            if(ngpus != 1) sync_nets(nets, ngpus, 0);

            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }

        if (i%10==0) {
            if (info) {
                //save info file frequently; close then reopen as appending
                fclose(infofile);
                infofile = fopen(infofile_name, "ab");
                fprintf(stderr, "Save training info to %s\n", infofile_name);
            }
        }

        if (i%10000==0 || (i < 1000 && i%100 == 0)) {

            if(ngpus != 1) sync_nets(nets, ngpus, 0);

            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }

    if(ngpus != 1) sync_nets(nets, ngpus, 0);

    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    if (info) {
        fclose(infofile);
        fprintf(stderr, "Training info: %s\n", infofile_name);
    }
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char* modules, char *outfile, NMS_MODE nms_mode)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    if (net->pre_transformable) {
        net->pre_transform = 1;
        pre_transform_conv_params(net);
    }


    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    fprintf(stderr, "\n");
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "\033[1A");
        fprintf(stderr, "\033[K");
        fprintf(stderr, "Validated: %d/%d\n", i, m);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms, nms_mode);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char* modules, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen, NMS_MODE nms_mode)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
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
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms, nms_mode);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

//================================================================================================================================================================================================================================
void test_list_detector(char *datacfg, char *cfgfile, char *weightfile, char* modules, char *filename, float thresh, float hier_thresh, char* prefix, int fullscreen, NMS_MODE nms_mode)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    char *valid_images = option_find_str(options, "valid", "data/train.list");
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    char outfile[256];
    float nms=.45;
    int i;
    printf("\n");

    for (i = 0; i < m; ++i) {
        // printf("\033[1A");
        // printf("\033[K");
        printf("Processed: %d/%d\n", i+1, m);
        image im = load_image_color(paths[i],0,0);
        sprintf(outfile, "%s%05d", prefix, i+1);
        
        image sized = letterbox_image(im, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = sized.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms, nms_mode);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);

        save_image(im, outfile);

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}

void make_box_gt(char *datacfg)
{
    int i,j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");

    char *boxgt_dir = option_find_str(options, "boxgt_dir", "results");

    int classes = option_find_int(options, "classes", 1);
    char **names = get_labels(name_list);
    char* label_dir = option_find_str_quiet(options, "label_dir", "labels");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;

    char buff[1024];
    FILE **fps = 0;

    fps = calloc(classes, sizeof(FILE *));
    for(i = 0; i < classes; ++i){
        snprintf(buff, 1024, "%s/box_gt_%s.txt", boxgt_dir, names[i]);
        fps[i] = fopen(buff, "w");
    }

    int w, h;

    printf("\n");

    for(i = 0; i < m; ++i) {
        printf("\033[1A");
        printf("\033[K");
        printf("Processed: %d/%d\n", i+1, m);
        char* path = paths[i];
        char* id = basecfg(path);

        image im = load_image(path,0,0,3);
        w = im.w;
        h = im.h;
        free_image(im);

        char labelpath[4096];
        find_replace(path, "images", label_dir, labelpath);
        find_replace(labelpath, "JPEGImages", label_dir, labelpath);
        find_replace(labelpath, ".png", ".txt", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        char labelpath_diff[4096];
        find_replace(labelpath, "labels", "labels_diff", labelpath_diff);
        FILE* diff_file = fopen(labelpath_diff, "r");
        int diff;

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);

        for(j = 0; j < num_labels; ++j){
            float xmin = truth[j].left * w;
            float xmax = truth[j].right * w;
            float ymin = truth[j].top * h;
            float ymax = truth[j].bottom * h;

            if (xmin < 1) xmin = 1;
            if (ymin < 1) ymin = 1;
            if (xmax > w) xmax = w;
            if (ymax > h) ymax = h;
            if (!diff_file) {
            	fprintf(fps[truth[j].id], "%s %d %f %f %f %f\n", id, 1, round(xmin), round(ymin), round(xmax), round(ymax));
            } else {
            	if (fscanf(diff_file, "%d", &diff) == 1) {
            		fprintf(fps[truth[j].id], "%s %d %f %f %f %f\n", id, 1 - diff, round(xmin), round(ymin), round(xmax), round(ymax));
            	} else {
            		fprintf(fps[truth[j].id], "%s %d %f %f %f %f\n", id, 1, round(xmin), round(ymin), round(xmax), round(ymax));
            	}
            }
        }
        if (diff_file) fclose(diff_file);
        free(id);
        free(truth);
    }

    for (i = 0; i < classes; ++i) {
        fclose(fps[i]);
    }
}

void eval(char *datacfg, float thresh)
{
	fprintf(stderr, "IMPORTANT: Make sure you run 'make_box_gt' first...\n");
    int i, j, k;
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");

    char *boxgt_dir = option_find_str(options, "boxgt_dir", "results");
    char *prefix = option_find_str(options, "results", "results");
    char *type = option_find_str(options, "eval", "voc2007");
    int classes = option_find_int(options, "classes", 1);
    char **names = get_labels(name_list);

    char buff[1024];
    FILE *fgt = 0;
    FILE *fdt = 0;

    //per-class eval
    float map = 0.0;
    fprintf(stderr, "total_dets \t tp/(tp + fn) \t recall \t precis \t AP \n");
    for (i = 0; i < classes; ++i) {
		char im_id[128];

		float p,l,t,r,b;

		int count = 0;
		int size = 64;
		im_box* gt_im_boxes = calloc(size, sizeof(im_box));
		float* gt_probs = calloc(size, sizeof(float));

        snprintf(buff, 1024, "%s/box_gt_%s.txt", boxgt_dir, names[i]);
        fgt = fopen(buff, "r");
    	if (!fgt) {
    		fprintf(stderr, "File error '%s'\n", buff);
            return;
    	}
    	
    	int npos = 0;

    	while (fscanf(fgt, "%s %f %f %f %f %f", im_id, &p, &l, &t, &r, &b) == 6) {
    		if (count == size) {
    			size *= 2;
    			gt_im_boxes = realloc(gt_im_boxes, size * sizeof(im_box));
    			gt_probs = realloc(gt_probs, size * sizeof(float));
    		}
    		strcpy(gt_im_boxes[count].id, im_id);
    		gt_im_boxes[count].bbox.x = l;
			gt_im_boxes[count].bbox.y = t;
			gt_im_boxes[count].bbox.w = r - l + 1;
			gt_im_boxes[count].bbox.h = b - t + 1;
			gt_probs[count] = p;
    		++count;
    		npos += (int)p;
    	}
    	int gt_total = count;

	    fclose(fgt);

    	count = 0;
    	size = 64;
		im_box* dt_im_boxes = calloc(size, sizeof(im_box));
		float* dt_probs = calloc(size, sizeof(float));

        snprintf(buff, 1024, "%s/comp4_det_test_%s.txt", prefix, names[i]);
        fdt = fopen(buff, "r");
    	if (!fdt) {
    		fprintf(stderr, "File error '%s'\n", buff);
            return;
    	}

    	while (fscanf(fdt, "%s %f %f %f %f %f", im_id, &p, &l, &t, &r, &b) == 6) {
            if (p < thresh) continue;
    		if (count == size) {
    			size *= 2;
    			dt_im_boxes = realloc(dt_im_boxes, size * sizeof(im_box));
    			dt_probs = realloc(dt_probs, size * sizeof(float));
    		}
    		strcpy(dt_im_boxes[count].id, im_id);
    		dt_im_boxes[count].bbox.x = l;
			dt_im_boxes[count].bbox.y = t;
			dt_im_boxes[count].bbox.w = r - l + 1;
			dt_im_boxes[count].bbox.h = b - t + 1;
			dt_probs[count] = -p;
    		++count;
    	}
    	int dt_total = count;
    	//nghiant_20190517: filter difficult things
    	int dt_total_nodiff = dt_total;
    	//nghiant_end

	    fclose(fdt);

	    int* sorted_pid = calloc(dt_total, sizeof(int));
	    for (j = 0; j < dt_total; ++j) {
	    	sorted_pid[j] = j;
	    }

	    merge_sort(dt_probs, 0, dt_total-1, sorted_pid);

	    int* tp = calloc(dt_total, sizeof(int));
	    int* fp = calloc(dt_total, sizeof(int));

	    for (j = 0; j < dt_total; ++j) {
	    	tp[j] = 0;
	    	fp[j] = 0;
	    }
	    
	    int* R = calloc(gt_total, sizeof(int));
		for (j = 0; j < gt_total; ++j) {
			R[j] = 0;
		}

	    int correct_det = 0;

		for (j = 0; j < dt_total; ++j) {
			int did = sorted_pid[j];
			box dtc_box = dt_im_boxes[did].bbox;
            
			float ovmax = 0;
			int gid_rec = -1;
			for (k = 0; k < gt_total; ++k) {
				if (strcmp(gt_im_boxes[k].id, dt_im_boxes[did].id) == 0) {
					box gtc_box = gt_im_boxes[k].bbox;

					//use custom iou instead
					float cl = dtc_box.x > gtc_box.x ? dtc_box.x : gtc_box.x;
					float ct = dtc_box.y > gtc_box.y ? dtc_box.y : gtc_box.y;
					float cr = (dtc_box.x + dtc_box.w - 1) < (gtc_box.x + gtc_box.w - 1) ? (dtc_box.x + dtc_box.w - 1) : (gtc_box.x + gtc_box.w - 1);
					float cb = (dtc_box.y + dtc_box.h - 1) < (gtc_box.y + gtc_box.h - 1) ? (dtc_box.y + dtc_box.h - 1) : (gtc_box.y + gtc_box.h - 1);
					float cw = (cr - cl + 1) > 0 ? (cr - cl + 1) : 0;
					float ch = (cb - ct + 1) > 0 ? (cb - ct + 1) : 0;
					float inters = cw * ch;
					float uni = dtc_box.w * dtc_box.h + gtc_box.w * gtc_box.h - inters;
					float overlaps = inters/uni;
					//use custom iou instead

					if (overlaps > 0.5 && overlaps > ovmax) {
						ovmax = overlaps;
						gid_rec = k;
					}
				}
			}

			if (gid_rec > -1) {
				if (gt_probs[gid_rec]) { //difficult images are skipped
					if (!R[gid_rec]) {
						tp[j] = 1;
						R[gid_rec] = 1;
						++correct_det;
					} else {
						fp[j] = 1;
					}
				} else {
					if (!R[gid_rec]) {
						--dt_total_nodiff;
					}
				}
			} else {
				fp[j] = 1;
			}
		}

	    for (j = 1; j < dt_total; ++j) {
	    	tp[j] += tp[j-1];
	    	fp[j] += fp[j-1];
	    }

		float ap = 0;
	    
	    if (strcmp(type,"voc2007") != 0) {
	    //VOC 2010 metric: TRUE METRIC
			float* precision = calloc(dt_total+2, sizeof(float));
			float* recall = calloc(dt_total+2, sizeof(float));
			precision[0] = 0.0;
			recall[0] = 0.0;
			precision[dt_total+1] = 0.0;
			recall[dt_total+1] = 1.0;
		    
		    for (j = 1; j < dt_total + 1; ++j) {
		    	recall[j] = (float)tp[j-1]/(float)npos;
		    	precision[j] = (float)tp[j-1]/(float)(tp[j-1] + fp[j-1]);
		    }

			for (j = dt_total + 1; j > 0; --j) {
				precision[j-1] = precision[j-1] > precision[j] ? precision[j-1] : precision[j];
			}

			for (j = 1; j < dt_total + 2; ++j) {
				if (recall[j] != recall[j-1]) {
					ap += (recall[j] - recall[j-1]) * precision[j];
				}
			}

			free(precision);
		    free(recall);
		} else {
		//VOC 2007 metric: 11-POINT METRIC
			float* precision = calloc(dt_total, sizeof(float));
			float* recall = calloc(dt_total, sizeof(float));
		    
		    for (j = 0; j < dt_total; ++j) {
		    	recall[j] = (float)tp[j]/(float)npos;
		    	precision[j] = (float)tp[j]/(float)(tp[j] + fp[j]);
		    }

			for (t = 0.0; t < 1.1; t = t + 0.1) {
				p = 0.0;
				for (j = 0; j < dt_total; ++j) {
					if (recall[j] >= t && precision[j] > p) {
						p = precision[j];
					}
				}
				ap = ap + p/11;
			}

			free(precision);
		    free(recall);
		}
		//END VOC 2007 METRIC

        fprintf(stderr, "%7d \t %5d/%5d \t %.4f \t %.4f \t AP for %s = %.4f\n", dt_total_nodiff, correct_det, npos, (float)(correct_det)/(npos), (float)(correct_det)/(dt_total_nodiff), names[i], ap);

		map = map + ap;

	    free(tp);
	    free(fp);
	    free(R);
	    free(sorted_pid);
	    free(gt_im_boxes);
		free(gt_probs);
		free(dt_im_boxes);
		free(dt_probs);
    }
    printf("Mean AP = %.4f\n", map/i);
}

void custom(char *datacfg, char *cfgfile, char *weightfile, char* modules, char *filename)
{
    int t = 0; //target layer for verification

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    char buff[256];
    char *input = buff;

    strncpy(input, filename, 256);
    image im = load_image_color(input,0,0);
    
    image sized = letterbox_image(im, net->w, net->h);

    //write to file
    FILE* inputfile = fopen("input_tensor.bin", "wb");
    fwrite(sized.data, sizeof(float), sized.w*sized.h*sized.c, inputfile);
    fclose(inputfile);
    //write to file

    layer l = net->layers[t];

    float *X = sized.data;
    network_predict(net, X);

    cuda_pull_array(l.output_gpu, l.output, l.outputs);
    float* Y = l.output;

    //write to file
    FILE* tensorfile = fopen("output_tensor.bin", "wb");
    fwrite(Y, sizeof(float), l.outputs, tensorfile);
    fclose(tensorfile);
    //write to file

    free_image(im);
    free_image(sized);
}

void custom2(char *datacfg, char *cfgfile, char *weightfile, char* modules, char *filename)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    float nms = .45;
    float thresh = 0.3;
    image **alphabet = load_alphabet();

    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    char buff[256];
    char *input = buff;

    strncpy(input, filename, 256);
    image im = load_image_color(input,0,0);
    
    //write to file
    FILE* oriinputfile = fopen("oriinput_tensor.bin", "wb");
    fwrite(im.data, sizeof(float), im.w*im.h*im.c, oriinputfile);
    fclose(oriinputfile);
    //write to file
    
    image sized = letterbox_image(im, net->w, net->h);
    
    //write to file
    FILE* inputfile = fopen("input_tensor.bin", "wb");
    fwrite(sized.data, sizeof(float), sized.w*sized.h*sized.c, inputfile);
    fclose(inputfile);
    //write to file

    layer l = net->layers[net->n-2];

    //read from file
    float* po = calloc(l.outputs, sizeof(float));
    FILE* ptoutput_file = fopen("/home/nghiant/git/policy_mask/ptoutput.bin", "rb");
    fread(po, sizeof(float), l.outputs, ptoutput_file);
    fclose(ptoutput_file);
    //read from file

    l.output = po;
    cuda_push_array(l.output_gpu, l.output, l.outputs);
    net->input_gpu = l.output_gpu;

    layer lx = net->layers[net->n-1];
    lx.forward_gpu(lx, *net);

    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, 0.5, 0, 1, &nboxes);
    if (nms) do_nms_sort(dets, nboxes, lx.classes, nms, NMS_NORMAL);
    draw_detections(im, dets, nboxes, thresh, names, alphabet, lx.classes);
    free_detections(dets, nboxes);
    fprintf(stderr, "%d\n", nboxes);

    save_image(im, "predictions");
    make_window("predictions", 512, 512, 0);
    show_image(im, "predictions", 0);

    free_image(im);
    free_image(sized);
}


void calc_anchor(char* datacfg, int c, int max_iters, int input_size) {
    fprintf(stderr, "Find %d anchors in %d iters\n", c, max_iters);
    int i,j;
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char* label_dir = option_find_str_quiet(options, "label_dir", "labels");

    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;

    float* data = calloc(2, sizeof(float));
    int n = 0;

    printf("\n");
    for(i = 0; i < m; ++i) {
        printf("\033[1A");
        printf("\033[K");
        printf("Explored: %d/%d\n", i+1, m);
        char* path = paths[i];

        char labelpath[4096];
        find_replace(path, "images", label_dir, labelpath);
        find_replace(labelpath, "JPEGImages", label_dir, labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        data = realloc(data, 2 * (n + num_labels) * sizeof(float));

        for(j = 0; j < num_labels; ++j){
            data[2*(n + j)]   = truth[j].w;
            data[2*(n + j)+1] = truth[j].h;
        }

        n += num_labels;
        
        free(truth);
    }

    float* centroid = calloc(2*c, sizeof(float));
    
    k_means(data, n/2, centroid, c, max_iters);
    
    for (i = 0; i < c; ++i) {
        printf("%f %f\n", input_size * centroid[2*i], input_size * centroid[2*i+1]);
    }

    free(centroid);
    free(data);
}

void create_binary_dataset(char *datacfg)
{
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    char filename[256];
    FILE *fp = 0;

    int m = plist->size;
    int i = 0;

    printf("\n");
    for (i = 0; i < m; ++i) {
        printf("\033[1A");
        printf("\033[K");
        printf("image: %d/%d\n", i+1, m);
        sprintf(filename, "voc_test_416x416_dnb/im_%d", i);
        fp = fopen(filename, "wb");
        image a = load_image_color(paths[i], 0, 0);
        image b = letterbox_image(a, 416, 416);

        fwrite(&(a.w), 1, sizeof(int), fp);
        fwrite(&(a.h), 1, sizeof(int), fp);
        fwrite(&(a.c), 1, sizeof(int), fp);
        fwrite(b.data, 416*416*3, sizeof(float), fp);

        fclose(fp);
        free_image(a);
        free_image(b);
    }
}

void run_detector(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s %s [train/test/valid]    --data [data_cfg] --config [cfg] --weight [weight (optional)] --module [module (optional)]\n", argv[0], argv[1]);
        fprintf(stderr, "       %s %s make_box_gt           --data [data_cfg]\n", argv[0], argv[1]);
        fprintf(stderr, "       %s %s eval                  --data [data_cfg]\n", argv[0], argv[1]);
        fprintf(stderr, "       %s %s anchor                --data [data_cfg] -anchor_n <num_anchor> -anchor_iters <max num iters> -anchor_size <input size>\n", argv[0], argv[1]);
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

    char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
    char *datacfg = find_char_2arg(argc, argv, "-d", "--data", 0);
    char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
    char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);

    if(0==strcmp(argv[2], "test")) {
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
        int nms_mode = find_int_arg(argc, argv, "-nms_mode", NMS_NORMAL);
        test_detector(datacfg, cfg, weights, modules, filename, thresh, hier_thresh, outfile, fullscreen, nms_mode);
    }
    else if(0==strcmp(argv[2], "test_list")) {
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        char *prefix = find_char_arg(argc, argv, "-prefix", 0);
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
        int nms_mode = find_int_arg(argc, argv, "-nms_mode", NMS_NORMAL);
        test_list_detector(datacfg, cfg, weights, modules, filename, thresh, hier_thresh, prefix, fullscreen, nms_mode);
    }
    else if(0==strcmp(argv[2], "train")) {
        int clear = find_arg(argc, argv, "-clear");
        int info = find_arg(argc, argv, "-info");
        train_detector(datacfg, cfg, weights, modules, gpus, ngpus, clear, info);
    }
    else if(0==strcmp(argv[2], "valid")) {
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        int nms_mode = find_int_arg(argc, argv, "-nms_mode", NMS_NORMAL);
        validate_detector(datacfg, cfg, weights, modules, outfile, nms_mode);
    }
    else if(0==strcmp(argv[2], "demo")) {
        int frame_skip = find_int_arg(argc, argv, "-skip", 0);
        int avg = find_int_arg(argc, argv, "-avg", 3);
        int width = find_int_arg(argc, argv, "-width", 0);
        int height = find_int_arg(argc, argv, "-height", 0);
        int fullscreen = find_arg(argc, argv, "-fullscreen");
        int cam_index = find_int_arg(argc, argv, "-cam", 0);
        int crop = find_int_arg(argc, argv, "-crop", 0);
        int save_frame = find_arg(argc, argv, "-save_frame");
        int fps = find_int_arg(argc, argv, "-fps", 0);
        char *prefix = find_char_arg(argc, argv, "-prefix", 0);
        char *filename = find_char_2arg(argc, argv, "-v", "--video", 0);
        float thresh = find_float_arg(argc, argv, "-thresh", .5);
        float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
        int add_frame_count = find_arg(argc, argv, "-count");
        char* line_150 = find_char_arg(argc, argv, "-line_150", 0);

        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, modules, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen, crop, save_frame, add_frame_count, line_150);
    }
    else if(0==strcmp(argv[2], "make_box_gt")) {
    	make_box_gt(datacfg);
    }
    else if(0==strcmp(argv[2], "eval")) {
        float thresh = find_float_arg(argc, argv, "-thresh", .005);
        eval(datacfg, thresh);
    }
    else if(0==strcmp(argv[2], "anchor")) {
        int anchor = find_int_arg(argc, argv, "-anchor_n", 5);
        int max_iters = find_int_arg(argc, argv, "-anchor_iters", 1000);
        int input_size = find_int_arg(argc, argv, "-anchor_size", 1);
        calc_anchor(datacfg, anchor, max_iters, input_size);
    }
    else if(0==strcmp(argv[2], "cdbn")) {
        create_binary_dataset(datacfg);
    }
    else if(0==strcmp(argv[2], "custom")) {
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        custom(datacfg, cfg, weights, modules, filename);
    }
    else if(0==strcmp(argv[2], "custom2")) {
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        custom2(datacfg, cfg, weights, modules, filename);
    }
}