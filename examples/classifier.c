#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, char* modules, int *gpus, int ngpus, int clear, int info)
{

    list *options = read_data_cfg(datacfg);
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *info_directory = option_find_str(options, "info", "/info/");
    char *label_path = option_find_str(options, "labels", 0);

    int classes = option_find_int(options, "classes", 2);

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
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);

    int imgs = net->batch * net->subdivisions * ngpus;

    char **labels = get_labels(label_path);
    list *plist = get_paths(train_list);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    double time;

    load_args args = get_base_args(net);

    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    fprintf(stderr, "%d %d\n", args.min, args.max);

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

    data train, buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;

    summarize_data_augmentation_options(args);

    FILE* infofile = 0;
    if (info) {
        infofile = fopen(infofile_name, "wb");
        save_training_info(infofile, net, 1, N);
    }
    fprintf(stderr, "Learning Rate: %g, Decay: %g\n", net->learning_rate, net->decay);

    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
            printf("Resizing\n");
            int dim = (rand() % 11 + 4) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
            printf("%d %d\n", args.min, args.max);

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
                //save info file frequently; close then reopen as appending
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
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void validate_classifier_crop(char *datacfg, char *filename, char *weightfile, char* modules)
{
    int i = 0;
    network *net = load_network(filename, weightfile, modules, 0, 0);

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile, char* modules)
{
    int i, j;
    network *net = load_network(filename, weightfile, modules, 0, 1);

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int w = net->w;
        int h = net->h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image_horizontal(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = calloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile, char* modules)
{
    int i, j;
    network *net = load_network(filename, weightfile, modules, 0, 1);

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    int size = net->w;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(net, resized.w, resized.h);
        float *pred = network_predict(net, resized.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_single(char *datacfg, char *filename, char *weightfile, char* modules, int batch)
{
    int i, j;
    if (batch <= 1) {
        fprintf(stderr, "Default validation: batch = 1\n");
        batch = 1;
    } else {
        fprintf(stderr, "Parallel validation: batch = %d\n", batch);
    }
    network *net = load_network(filename, weightfile, modules, 0, batch);

    if (net->pre_transformable) {
        net->pre_transform = 1;
        pre_transform_conv_params(net);
    }

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));
    fprintf(stderr, "\n");

    double start = what_time_is_it_now();

    if (batch == 1) {
        for(i = 0; i < m; ++i){
            int class = -1;
            char *path = paths[i];
            
            char* imagename = basecfg(path);
            int label_length = 0; //select longest string, guarantee subtring in string

            for(j = 0; j < classes; ++j){
                if(strstr(imagename, labels[j])){
                    if (strlen(labels[j]) > label_length) {
                        label_length = strlen(labels[j]);
                        class = j;
                    }
                }
            }
            free(imagename);
            image im = load_image_color(paths[i], 0, 0);
            image crop = center_crop_image(im, net->w, net->h);

            float *pred = network_predict(net, crop.data);
            if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

            free_image(im);
            free_image(crop);
            top_k(pred, classes, topk, indexes);

            if(indexes[0] == class) avg_acc += 1;
            for(j = 0; j < topk; ++j){
                if(indexes[j] == class) avg_topk += 1;
            }
            fprintf(stderr, "\033[1A");
            fprintf(stderr, "\033[K");
            fprintf(stderr, "%6d/%6d: top 1: %f, top %d: %f\n", i+1, m, avg_acc/(i+1), topk, avg_topk/(i+1));
        }
    } else {
        load_args args = {0};
        args.hierarchy = net->hierarchy;

        args.w = net->w;
        args.h = net->h;
        args.type = IMAGE_DATA_CROP;
        args.labels = labels;
        args.classes = classes;

        image* val = calloc(batch, sizeof(image));
        image* val_resized = calloc(batch, sizeof(image));
        image* buf = calloc(batch, sizeof(image));
        image* buf_resized = calloc(batch, sizeof(image));

        float* buf_truth = calloc(classes*batch, sizeof(float));
        float* X = calloc(net->w*net->h*3*batch, sizeof(float));
        int* y = calloc(batch, sizeof(int));

        pthread_t *thr = calloc(batch, sizeof(pthread_t));
        i = 0;
        int t;
        for (t = 0; t < batch; ++t) {
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            args.truth = buf_truth+t*classes;
            thr[t] = load_data_in_thread(args);
        }
        for (i = batch; i < m+batch; i += batch) {
            for (t = 0; t < batch && i+t-batch < m; ++t) {
                fprintf(stderr, "\033[1A");
                fprintf(stderr, "\033[K");
                fprintf(stderr, "Validated: %d/%d\n", i+t-batch+1, m);
                pthread_join(thr[t], 0);
                val[t] = buf[t];
                val_resized[t] = buf_resized[t];
                memcpy(X + t*net->w*net->h*3, val_resized[t].data, net->w*net->h*3*sizeof(float));
                for (j = 0; j < classes; ++j) {
                    if (buf_truth[t*classes+j] == 1) {
                        y[t] = j;
                    }
                }
            }

            for (t = 0; t < batch && i+t < m; ++t) {
                args.path = paths[i+t];
                args.im = &buf[t];
                args.resized = &buf_resized[t];
                args.truth = buf_truth+t*classes;
                thr[t] = load_data_in_thread(args);
            }
            float* pred = network_predict(net, X);
            for (t = 0; t < batch && i+t-batch < m; ++t) {
                top_k(pred+t*classes, classes, topk, indexes);

                if(indexes[0] == y[t]) avg_acc += 1;
                for(j = 0; j < topk; ++j){
                    if(indexes[j] == y[t]) avg_topk += 1;
                }
                free_image(val[t]);
                free_image(val_resized[t]);
            }

        }
        // fprintf(stderr, "top 1: %f, top %d: %f\n", avg_acc/m, topk, avg_topk/m);
        printf("top 1: %f, top %d: %f\n", avg_acc/m, topk, avg_topk/m);
    }
    fprintf(stderr, "Total Classification Time: %f Seconds\n", what_time_is_it_now() - start);
}

void validate_classifier_multi(char *datacfg, char *cfg, char *weights, char* modules)
{
    int i, j;
    network *net = load_network(cfg, weights, modules, 0, 1);

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    //int scales[] = {224, 288, 320, 352, 384};
    int scales[] = {224, 256, 288, 320};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_max(im, scales[j]);
            resize_network(net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image_horizontal(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char* modules, char *filename, int layer_num)
{
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net->layers[layer_num];
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }

        cuda_pull_array(l.output_gpu, l.output, l.outputs);

        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }

        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char* modules, char *filename, int top)
{
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = letterbox_image(im, net->w, net->h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}


void label_classifier(char *datacfg, char *filename, char *weightfile, char* modules)
{
    int i;
    network *net = load_network(filename, weightfile, modules, 0, 1);

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);

        printf("%s\n", labels[ind]);
    }
}

void csv_classifier(char *datacfg, char *cfgfile, char *weightfile, char* modules)
{
    int i,j;
    network *net = load_network(cfgfile, weightfile, modules, 0, 0);

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int top = option_find_int(options, "top", 1);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    int *indexes = calloc(top, sizeof(int));

    for(i = 0; i < m; ++i){
        double time = what_time_is_it_now();
        char *path = paths[i];
        image im = load_image_color(path, 0, 0);
        image r = letterbox_image(im, net->w, net->h);
        float *predictions = network_predict(net, r.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);

        printf("%s", path);
        for(j = 0; j < top; ++j){
            printf("\t%d", indexes[j]);
        }
        printf("\n");

        free_image(im);
        free_image(r);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", what_time_is_it_now() - time, i+1, m);
    }
}

void test_classifier(char *datacfg, char *cfgfile, char *weightfile, char* modules, int target_layer)
{
    int curr = 0;
    network *net = load_network(cfgfile, weightfile, modules, 0, 0);

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.classes = classes;
    args.n = net->batch;
    args.m = 0;
    args.labels = 0;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(curr = net->batch; curr < m; curr += net->batch){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        if(curr < m){
            args.paths = paths + curr;
            if (curr + net->batch > m) args.n = m - curr;
            load_thread = load_data_in_thread(args);
        }
        fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);

        int i, j;

        for(i = 0; i < pred.rows; ++i){
            printf("%s", paths[curr-net->batch+i]);
            for(j = 0; j < pred.cols; ++j){
                printf("\t%g", pred.vals[i][j]);
            }
            printf("\n");
        }

        free_matrix(pred);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr);
        free_data(val);
    }
}

void file_output_classifier(char *datacfg, char *filename, char *weightfile, char* modules, char *listfile)
{
    int i,j;
    network *net = load_network(filename, weightfile, modules, 0, 1);

    list *options = read_data_cfg(datacfg);

    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(listfile);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);

        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 0, 1);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);

        printf("%s", paths[i]);
        for(j = 0; j < classes; ++j){
            printf("\t%g", pred[j]);
        }
        printf("\n");
    }
}

void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, char* modules, int cam_index, const char *filename)
{
    char *base = basecfg(cfgfile);
    image **alphabet = load_alphabet();
    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);
    list *options = read_data_cfg(datacfg);


    int w = 1280;
    int h = 720;
    void * cap = open_video_stream(filename, cam_index, w, h, 0);

    int top = option_find_int(options, "top", 1);

    char *label_list = option_find_str(options, "labels", 0);
    char *name_list = option_find_str(options, "names", label_list);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);

        float *predictions = network_predict(net, in_s.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        int lh = in.h*.03;
        int toph = 3*lh;

        float rgb[3] = {1,1,1};
        for(i = 0; i < top; ++i){
            printf("%d\n", toph);
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);

            char buff[1024];
            sprintf(buff, "%3.1f%%: %s\n", predictions[index]*100, names[index]);
            image label = get_label(alphabet, buff, lh);
            draw_label(in, toph, lh, label, rgb);
            toph += 2*lh;
            free_image(label);
        }

        show_image(in, base, 10);
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
}


void run_classifier(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);

    char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
    char *data = find_char_2arg(argc, argv, "-d", "--data", 0);
    char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
    char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);

    if(0==strcmp(argv[2], "predict")) {
        int top = find_int_arg(argc, argv, "-t", 0);
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        predict_classifier(data, cfg, weights, modules, filename, top);
    }
    
    else if(0==strcmp(argv[2], "fout")) {
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        file_output_classifier(data, cfg, weights, modules, filename);
    }

    else if(0==strcmp(argv[2], "try")) {
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        int layer = find_int_arg(argc, argv, "-layer", 0);
        try_classifier(data, cfg, weights, modules, filename, layer);
    }

    else if(0==strcmp(argv[2], "train")) {
        int info = find_arg(argc, argv, "-info");
        int clear = find_arg(argc, argv, "-clear");
        train_classifier(data, cfg, weights, modules, gpus, ngpus, clear, info);
    }

    else if(0==strcmp(argv[2], "demo")) {
        int cam_index = find_int_arg(argc, argv, "-cam", 0);
        char *filename = find_char_2arg(argc, argv, "-v", "--video", 0);
        demo_classifier(data, cfg, weights, modules, cam_index, filename);
    }

    else if(0==strcmp(argv[2], "test")) {
        int layer = find_int_arg(argc, argv, "-layer", -1);
        test_classifier(data, cfg, weights, modules, layer);
    }

    else if(0==strcmp(argv[2], "csv")) {
        csv_classifier(data, cfg, weights, modules);
    }

    else if(0==strcmp(argv[2], "label")) {
        label_classifier(data, cfg, weights, modules);
    }

    else if(0==strcmp(argv[2], "valid")) {
        int batch = find_int_arg(argc, argv, "-batch", 1);
        validate_classifier_single(data, cfg, weights, modules, batch);
    }

    else if(0==strcmp(argv[2], "validmulti")) {
        validate_classifier_multi(data, cfg, weights, modules);
    }

    else if(0==strcmp(argv[2], "valid10")) {
        validate_classifier_10(data, cfg, weights, modules);
    }

    else if(0==strcmp(argv[2], "validcrop")) {
        validate_classifier_crop(data, cfg, weights, modules);
    }

    else if(0==strcmp(argv[2], "validfull")) {
        validate_classifier_full(data, cfg, weights, modules);
    }

}


