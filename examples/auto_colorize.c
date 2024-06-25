#include "darknet.h"
#include "utils.h"
#include "image.h"
void train_auto_colorizer(char *datacfg, char *cfgfile, char *weightfile, char* modules, int *gpus, int ngpus, int clear, int info)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *info_directory = option_find_str(options, "info", "/info/");

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

    list *plist = get_paths(train_images);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = get_base_args(net);

    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.d = &buffer;
    args.type = AUTO_COLORIZE_DATA;
    args.threads = 64;

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

void test_auto_colorizer(char *cfgfile, char *weightfile, char* modules, char *filename, char* outfile)
{
    network *net = load_network(cfgfile, weightfile, modules, 0, 1);

    char buff[256];
    char *input = buff;
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
        image sized = resize_image(im, net->w, net->h);
        rgb_to_lab(sized);

        image clone = copy_image(sized);

        float *X = sized.data;
        float* ab_channel = network_predict(net, X);
        memcpy(clone.data + net->w*net->h, ab_channel, net->w * net->h * 2 * sizeof(float));
        lab_to_rgb(clone);
        image orig_clone = resize_image(clone, im.w, im.h);


        if(outfile){
            save_image(orig_clone, outfile);
        }
        else{
            save_image(orig_clone, "colorize");
            make_window("colorize", im.w, im.h, 0);
            show_image(orig_clone, "colorize", 0);
        }

        free_image(im);
        free_image(sized);
        free_image(clone);
        free_image(orig_clone);
        if (filename) break;
    }
}

void run_auto_colorizer(int argc, char **argv)
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

    char *modules = find_char_2arg(argc, argv, "-m", "--module", 0);
    char *datacfg = find_char_2arg(argc, argv, "-d", "--data", 0);
    char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
    char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
    
    if (0==strcmp(argv[2], "train")) {
        int clear = find_arg(argc, argv, "-clear");
        int info = find_arg(argc, argv, "-info");
        train_auto_colorizer(datacfg, cfg, weights, modules, gpus, ngpus, clear, info);
    }
    else if (0==strcmp(argv[2], "test")) {
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        test_auto_colorizer(cfg, weights, modules, filename, outfile);
    }
}
