#include "darknet.h"
#include "utils.h"
#include "image.h"

void test_super_resolution(char *cfgfile, char *weightfile, char* modules, char *filename, char* outfile)
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
        fprintf(stderr, "%d %d\n", im.w, im.h);
        image simple_resize = resize_image(im, net->w/2, net->h/2);
        simple_resize = resize_image(simple_resize, net->w, net->h);
        
        image vdsr_res = copy_image(simple_resize);

        rgb_to_ycbcr(vdsr_res);

        float *X = vdsr_res.data;
        float* res_im = network_predict(net, X);
        memcpy(vdsr_res.data, res_im, net->w * net->h * sizeof(float));
        ycbcr_to_rgb(vdsr_res);
        constrain_image(vdsr_res);

        if(outfile){
            save_image(vdsr_res, outfile);
        }
        else{
            save_image(simple_resize, "upsa");
            save_image(vdsr_res, "sres");
            make_window("colorize", vdsr_res.w, vdsr_res.h, 0);
            show_image(vdsr_res, "colorize", 0);
        }

        free_image(im);
        free_image(simple_resize);
        free_image(vdsr_res);
        if (filename) break;
    }
}

void run_super_resolution(int argc, char **argv)
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
    char *cfg = find_char_2arg(argc, argv, "-c", "--config", 0);
    char *weights = find_char_2arg(argc, argv, "-w", "--weight", 0);
    
    if (0==strcmp(argv[2], "test")) {
        char *filename = find_char_2arg(argc, argv, "-im", "--image", 0);
        char *outfile = find_char_arg(argc, argv, "-out", 0);
        test_super_resolution(cfg, weights, modules, filename, outfile);
    } else {
        fprintf(stderr, "none binary found\n");
    }
}
