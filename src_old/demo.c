#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static network *net;
static image buff [3];
static image buff_letter[3];

static int buff_index = 0;
static void * cap;
static void* vwriter;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_index = 0;
static float **predictions;
static float *avg;
static int demo_done = 0;
static int demo_total = 0;
static int crop = 0;
double demo_time;

int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}

void remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}

detection *avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(demo_total, 0, avg, 1);
    for(j = 0; j < demo_frame; ++j){
        axpy_cpu(demo_total, 1./demo_frame, predictions[j], 1, avg, 1);
    }
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, buff[0].w, buff[0].h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}

void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .45;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    network_predict(net, X);

    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    dets = avg_predictions(net, &nboxes);

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = buff[(buff_index+2) % 3];
    draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    free_detections(dets, nboxes);

    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    free_image(buff[buff_index]);
    buff[buff_index] = get_image_from_stream(cap);
    if(buff[buff_index].data == 0) {
        demo_done = 1;
        return 0;
    }
    if (crop) {
        if (crop > buff[0].w || crop > buff[0].h) {
            crop = buff[0].w < buff[0].h ? buff[0].w : buff[0].h;
        }
        int dx = (buff[buff_index].w - crop) / 2;
        int dy = (buff[buff_index].h - crop) / 2;
        buff[buff_index] = crop_image(buff[buff_index], dx, dy, crop, crop);
    }
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    return 0;
}

void *display_in_thread(void *ptr)
{
    int c = show_image(buff[(buff_index + 1)%3], "Demo", 1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void demo(char *cfgfile, char *weightfile, char* modules, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, int cropx, int save_frame, int add_frame_count, char* line_150)
{
    image **alphabet = load_alphabet();
    demo_frame = avg_frames;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    crop = cropx;
    printf("Demo\n");
    net = load_network(cfgfile, weightfile, modules, 0, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;


    int i;
    demo_total = size_network(net);
    predictions = calloc(demo_frame, sizeof(float*));
    for (i = 0; i < demo_frame; ++i){
        predictions[i] = calloc(demo_total, sizeof(float));
    }
    avg = calloc(demo_total, sizeof(float));

    if(filename){
        printf("video file: %s\n", filename);
        cap = open_video_stream(filename, 0, 0, 0, 0);
    }else{
        cap = open_video_stream(0, cam_index, w, h, frames);
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    int count = 0;
    if(!prefix){
        make_window("Demo", 1352, 1013, fullscreen);
    } else {
        char output_video[256];
        sprintf(output_video, "%s.avi", prefix);
        if (crop) {
            if (crop > buff[0].w || crop > buff[0].h) {
                crop = buff[0].w < buff[0].h ? buff[0].w : buff[0].h;
            }
            vwriter = open_video_writer(output_video, crop, crop, frames);
        } else {
            vwriter = open_video_writer(output_video, buff[0].w, buff[0].h, frames);
        }
        fprintf(stderr, "Saving results to %s @ %f\n", output_video, (float)frames);
    }

    demo_time = what_time_is_it_now();
    int frame_count = 0;

    while(!demo_done){
        ++frame_count;
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }else if (save_frame) {
            char name[256];
            sprintf(name, "%s_%08d", prefix, frame_count);
            if (add_frame_count) {
                char info[256];
                sprintf(info, "%6d", frame_count);
                image label = get_label(alphabet, info, (buff[0].h*.01));
                float rgb[3] = {0.2,1.0,0.2};
                draw_label(buff[(buff_index + 1)%3], 10, 10, label, rgb);
                free_image(label);
            }
            save_image(buff[(buff_index + 1)%3], name);
        } else {
            if (line_150) {
                FILE* line_150_file = fopen(line_150, "r");
                int x1, y1, x2, y2;
                fscanf(line_150_file, "%d %d %d %d", &x1, &y1, &x2, &y2);
                draw_line(&buff[(buff_index + 1)%3], x1, y1, x2, y2, 20, 20, 250, 3);
            }

            if (add_frame_count) {
                char info[256];
                sprintf(info, "%6d", frame_count);
                image label = get_label(alphabet, info, (buff[0].h*.01));
                float rgb[3] = {0.2,1.0,0.2};
                draw_label(buff[(buff_index + 1)%3], 10, 10, label, rgb);
                free_image(label);
            }

            record_video(vwriter, buff[(buff_index + 1)%3]);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
    if(!prefix) release_video_writer(vwriter);
    fprintf(stderr, "Frame counter: %d\n", frame_count);
}