#include "data.h"
#include "utils.h"
#include "image.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void summarize_data_augmentation_options(load_args args) {
    fprintf(stderr, "===Augmentation===\n");
    if (args.angle != 0)                       {fprintf(stderr, "\033[0;32mRotate      %6.2f\033[0m\n", args.angle);}
    else                                       {fprintf(stderr, "\033[0;31mRotate      %6.2f\033[0m\n", args.angle);}

    if (args.jitter != 0)                      {fprintf(stderr, "\033[0;32mJitter      %6.2f\033[0m\n", args.jitter);}
    else                                       {fprintf(stderr, "\033[0;31mJitter      %6.2f\033[0m\n", args.jitter);}
    
    if (args.saturation != 1)                  {fprintf(stderr, "\033[0;32mSaturation  %6.2f\033[0m\n", args.saturation);}
    else                                       {fprintf(stderr, "\033[0;31mSaturation  %6.2f\033[0m\n", args.saturation);}
    
    if (args.exposure != 1)                    {fprintf(stderr, "\033[0;32mExposure    %6.2f\033[0m\n", args.exposure);}
    else                                       {fprintf(stderr, "\033[0;31mExposure    %6.2f\033[0m\n", args.exposure);}
    
    if (args.hue != 0)                         {fprintf(stderr, "\033[0;32mHue         %6.2f\033[0m\n", args.hue);}
    else                                       {fprintf(stderr, "\033[0;31mHue         %6.2f\033[0m\n", args.hue);}
    
    if (args.zoom[0] < 1 || args.zoom[1] < 1)  {fprintf(stderr, "\033[0;32mZoom Out\033[0m\n");}
    else                                       {fprintf(stderr, "\033[0;31mZoom Out\033[0m\n");}
    
    if (args.zoom[0] > 1 || args.zoom[1] > 1)  {fprintf(stderr, "\033[0;32mZoom In \033[0m\n");}
    else                                       {fprintf(stderr, "\033[0;31mZoom In \033[0m\n");}

                                               {fprintf(stderr, "\033[0;33mRange: %4.2f ~ %4.2f\033[0m\n", args.zoom[0] < args.zoom[1] ? args.zoom[0] : args.zoom[1], args.zoom[0] < args.zoom[1] ? args.zoom[1] : args.zoom[0]);}
    
    if (args.hflip != 0)                       {fprintf(stderr, "\033[0;32mH-Flip      %6.2f\033[0m\n", args.hflip);}
    else                                       {fprintf(stderr, "\033[0;31mH-Flip      %6.2f\033[0m\n", args.hflip);}
    
    if (args.vflip != 0)                       {fprintf(stderr, "\033[0;32mV-Flip      %6.2f\033[0m\n", args.vflip);}
    else                                       {fprintf(stderr, "\033[0;31mV-Flip      %6.2f\033[0m\n", args.vflip);}
    
    if (args.solarize != 0)                    {fprintf(stderr, "\033[0;32mSolarize    %6.2f\033[0m\n", args.solarize);}
    else                                       {fprintf(stderr, "\033[0;31mSolarize    %6.2f\033[0m\n", args.solarize);}
    
    if (args.posterize != 0)                   {fprintf(stderr, "\033[0;32mPosterize   %6.2f\033[0m\n", args.posterize);}
    else                                       {fprintf(stderr, "\033[0;31mPosterize   %6.2f\033[0m\n", args.posterize);}
    
    if (args.noise != 0)                       {fprintf(stderr, "\033[0;32mNoise       %6.2f\033[0m\n", args.noise);}
    else                                       {fprintf(stderr, "\033[0;31mNoise       %6.2f\033[0m\n", args.noise);}

    if (args.cutout.prob != 0) {
    							               fprintf(stderr, "\033[0;32mCutout      %6.2f\033[0m\n", args.cutout.prob);
                                               fprintf(stderr, "\033[0;33mSize:  %4d x %4d\033[0m\n", args.cutout.max_w, args.cutout.max_h);
	} else {
											   fprintf(stderr, "\033[0;31mCutout      %6.2f\033[0m\n", args.cutout.prob);
                                               fprintf(stderr, "\033[0;33mSize:      - x   -\033[0m\n");
	}

    
    fprintf(stderr, "==================\n");
}

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop;
        if(center){
            crop = center_crop_image(im, size, size);
        } else {
            crop = random_augment_image(im, angle, aspect, min, max, size, size);
        }
        int flip_h = (rand() % 100) < (int)(hflip * 100.);
        int flip_v = (rand() % 100) < (int)(vflip * 100.);
        flip_image_x(crop, flip_h, flip_v);

        random_distort_image(crop, hue, saturation, exposure);
        random_distort_image_extend(crop, solarize, posterize, noise);
        random_cutout_image(crop, cutout);

        free_image(im);
        X.vals[i] = crop.data;
        X.cols = crop.h*crop.w*crop.c;
    }

    return X;
}

box_label *read_boxes(char *filename, int *n)
{
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    int size = 64;
    box_label *boxes = calloc(size, sizeof(box_label));
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        if(count == size) {
            size = size * 2;
            boxes = realloc(boxes, size*sizeof(box_label));
        }
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip_h, int flip_v)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip_h){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        if(flip_v){
            float swap = boxes[i].top;
            boxes[i].top = 1. - boxes[i].bottom;
            boxes[i].bottom = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void load_rle(image im, int *rle, int n)
{
    int count = 0;
    int curr = 0;
    int i,j;
    for(i = 0; i < n; ++i){
        for(j = 0; j < rle[i]; ++j){
            im.data[count++] = curr;
        }
        curr = 1 - curr;
    }
    for(; count < im.h*im.w*im.c; ++count){
        im.data[count] = curr;
    }
}

void or_image(image src, image dest, int c)
{
    int i;
    for(i = 0; i < src.w*src.h; ++i){
        if(src.data[i]) dest.data[dest.w*dest.h*c + i] = 1;
    }
}

box bound_image(image im)
{
    int x,y;
    int minx = im.w;
    int miny = im.h;
    int maxx = 0;
    int maxy = 0;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            if(im.data[y*im.w + x]){
                minx = (x < minx) ? x : minx;
                miny = (y < miny) ? y : miny;
                maxx = (x > maxx) ? x : maxx;
                maxy = (y > maxy) ? y : maxy;
            }
        }
    }
    box b = {minx, miny, maxx-minx + 1, maxy-miny + 1};
    return b;
}

void fill_truth_iseg(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip_h, int flip_v, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    int j;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);

        flip_image_x(sized, flip_h, flip_v);

        image mask = resize_image(sized, mw, mh);
        truth[i*(mw*mh+1)] = id;
        for(j = 0; j < mw*mh; ++j){
            truth[i*(mw*mh + 1) + 1 + j] = mask.data[j];
        }
        ++i;

        free_image(mask);
        free_image(sized);
        free(rle);
    }
    if(i < num_boxes) truth[i*(mw*mh+1)] = -1;
    fclose(file);
    free_image(part);
}

void fill_truth_mask(char *path, int num_boxes, float *truth, int classes, int w, int h, augment_args aug, int flip_h, int flip_v, int mw, int mh)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    int i = 0;
    image part = make_image(w, h, 1);
    while((fscanf(file, "%d %s", &id, buff) == 2) && i < num_boxes){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        image sized = rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);

        flip_image_x(sized, flip_h, flip_v);

        box b = bound_image(sized);
        if(b.w > 0){
            image crop = crop_image(sized, b.x, b.y, b.w, b.h);
            image mask = resize_image(crop, mw, mh);
            truth[i*(4 + mw*mh + 1) + 0] = (b.x + b.w/2.)/sized.w;
            truth[i*(4 + mw*mh + 1) + 1] = (b.y + b.h/2.)/sized.h;
            truth[i*(4 + mw*mh + 1) + 2] = b.w/sized.w;
            truth[i*(4 + mw*mh + 1) + 3] = b.h/sized.h;
            int j;
            for(j = 0; j < mw*mh; ++j){
                truth[i*(4 + mw*mh + 1) + 4 + j] = mask.data[j];
            }
            truth[i*(4 + mw*mh + 1) + 4 + mw*mh] = id;
            free_image(crop);
            free_image(mask);
            ++i;
        }
        free_image(sized);
        free(rle);
    }
    fclose(file);
    free_image(part);
}

void fill_truth_detection(char *path, char* label_dir, int num_boxes, float *truth, int classes, float rad, int iw, int ih, int rw, int rh, int flip_h, int flip_v, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];

    find_replace(path, "images", label_dir, labelpath);
    find_replace(labelpath, "JPEGImages", label_dir, labelpath);
    find_replace(labelpath, "raw", label_dir, labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".png", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    int i;

    //nghiant: correct boxes of rotated image
    float max_x, max_y;
    float min_x, min_y;
    float cos_rad = cos(rad);
    float sin_rad = sin(rad);

    float tl_x = 0;
    float tl_y = 0;

    float tr_x =  cos_rad * iw;
    float tr_y = -sin_rad * iw;
    
    float br_x =  cos_rad * iw + sin_rad * ih;
    float br_y = -sin_rad * iw + cos_rad * ih;
    
    float bl_x = sin_rad * ih;
    float bl_y = cos_rad * ih;

    find_min_max(tl_x, tr_x, br_x, bl_x, &min_x, &max_x);
    find_min_max(tl_y, tr_y, br_y, bl_y, &min_y, &max_y);

    for (i = 0; i < count; ++i) {
    	float bl = boxes[i].left   * iw;
    	float br = boxes[i].right  * iw;
    	float bt = boxes[i].top    * ih;
    	float bb = boxes[i].bottom * ih;

    	float tmax_x, tmin_x, tmax_y, tmin_y;
        float ttl_tx =  cos_rad * bl + sin_rad * bt;
        float ttl_ty = -sin_rad * bl + cos_rad * bt;

        float ttr_tx =  cos_rad * br + sin_rad * bt;
        float ttr_ty = -sin_rad * br + cos_rad * bt;

        float tbr_tx =  cos_rad * br + sin_rad * bb;
        float tbr_ty = -sin_rad * br + cos_rad * bb;
        
        float tbl_tx =  cos_rad * bl + sin_rad * bb;
        float tbl_ty = -sin_rad * bl + cos_rad * bb;

        find_min_max(ttl_tx, ttr_tx, tbr_tx, tbl_tx, &tmin_x, &tmax_x);
        find_min_max(ttl_ty, ttr_ty, tbr_ty, tbl_ty, &tmin_y, &tmax_y);

        bl = (tmin_x - min_x) / rw;
        br = (tmax_x - min_x) / rw;
        bt = (tmin_y - min_y) / rh;
        bb = (tmax_y - min_y) / rh;

        boxes[i].left   = bl;
		boxes[i].right  = br;
		boxes[i].top    = bt;
		boxes[i].bottom = bb;

        boxes[i].x = (bl + br) / 2;
		boxes[i].y = (bt + bb) / 2;
		boxes[i].w = br - bl;
		boxes[i].h = bb - bt;
    }
    
    //nghiant_end

    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip_h, flip_v);
    if(count > num_boxes) count = num_boxes;
    float x,y,w,h;
    int id;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if ((w < .001 || h < .001)) {
            ++sub;
            continue;
        }

        truth[(i-sub)*5+0] = x;
        truth[(i-sub)*5+1] = y;
        truth[(i-sub)*5+2] = w;
        truth[(i-sub)*5+3] = h;
        truth[(i-sub)*5+4] = id;
    }
    free(boxes);
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    
    char* filename = basecfg(path);
    int label_length = 0; //select longest string, guarantee subtring in string
    int label_index = -1;

    for(i = 0; i < k; ++i){
        if(strstr(filename, labels[i])){
            if (strlen(labels[i]) > label_length) {
                label_length = strlen(labels[i]);
                label_index = i;
            }
        }
    }
    truth[label_index] = 1;
    free(filename);
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

matrix load_regression_labels_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i,j;
    for(i = 0; i < n; ++i){
        char labelpath[4096];
        find_replace(paths[i], "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".BMP", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);
        find_replace(labelpath, ".JPG", ".txt", labelpath);
        find_replace(labelpath, ".JPeG", ".txt", labelpath);
        find_replace(labelpath, ".Jpeg", ".txt", labelpath);
        find_replace(labelpath, ".PNG", ".txt", labelpath);
        find_replace(labelpath, ".TIF", ".txt", labelpath);
        find_replace(labelpath, ".bmp", ".txt", labelpath);
        find_replace(labelpath, ".jpeg", ".txt", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".png", ".txt", labelpath);
        find_replace(labelpath, ".tif", ".txt", labelpath);

        FILE *file = fopen(labelpath, "r");
        for(j = 0; j < k; ++j){
            fscanf(file, "%f", &(y.vals[i][j]));
        }
        fclose(file);
    }
    return y;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
        if(hierarchy){
            fill_hierarchy(y.vals[i], k, hierarchy);
        }
    }
    return y;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char **get_labels_with_n(char *filename, int* n)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    *n = plist->size;
    free_list(plist);
    return labels;
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

image get_segmentation_image(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes);
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        free(rle);
    }
    fclose(file);
    free_image(part);
    return mask;
}

image get_segmentation_image2(char *path, int w, int h, int classes)
{
    char labelpath[4096];
    find_replace(path, "images", "mask", labelpath);
    find_replace(labelpath, "JPEGImages", "mask", labelpath);
    find_replace(labelpath, ".jpg", ".txt", labelpath);
    find_replace(labelpath, ".JPG", ".txt", labelpath);
    find_replace(labelpath, ".JPEG", ".txt", labelpath);
    image mask = make_image(w, h, classes+1);
    int i;
    for(i = 0; i < w*h; ++i){
        mask.data[w*h*classes + i] = 1;
    }
    FILE *file = fopen(labelpath, "r");
    if(!file) file_error(labelpath);
    char buff[32788];
    int id;
    image part = make_image(w, h, 1);
    while(fscanf(file, "%d %s", &id, buff) == 2){
        int n = 0;
        int *rle = read_intlist(buff, &n, 0);
        load_rle(part, rle, n);
        or_image(part, mask, id);
        for(i = 0; i < w*h; ++i){
            if(part.data[i]) mask.data[w*h*classes + i] = 0;
        }
        free(rle);
    }
    fclose(file);
    free_image(part);
    return mask;
}

data load_data_seg(int n, char **paths, int m, int w, int h, int classes, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, int div, float hflip, float vflip, float solarize, float posterize, float noise)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    d.y.rows = n;
    d.y.cols = h*w*classes/div/div;
    d.y.vals = calloc(d.X.rows, sizeof(float*));

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip_h = (rand() % 100) < (int)(hflip * 100.);
        int flip_v = (rand() % 100) < (int)(vflip * 100.);

        flip_image_x(sized, flip_h, flip_v);

        random_distort_image(sized, hue, saturation, exposure);
        random_distort_image_extend(sized, solarize, posterize, noise);
        d.X.vals[i] = sized.data;

        image mask = get_segmentation_image(random_paths[i], orig.w, orig.h, classes);
        image sized_m = rotate_crop_image(mask, a.rad, a.scale/div, a.w/div, a.h/div, a.dx/div, a.dy/div, a.aspect);

        flip_image_x(sized_m, flip_h, flip_v);

        d.y.vals[i] = sized_m.data;

        free_image(orig);
        free_image(mask);
    }
    free(random_paths);
    return d;
}

data load_data_iseg(int n, char **paths, int m, int w, int h, int classes, int boxes, int div, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, float hflip, float vflip, float solarize, float posterize, float noise)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (((w/div)*(h/div))+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip_h = (rand() % 100) < (int)(hflip * 100.);
        int flip_v = (rand() % 100) < (int)(vflip * 100.);
        flip_image_x(sized, flip_h, flip_v);

        random_distort_image(sized, hue, saturation, exposure);
        random_distort_image_extend(sized, solarize, posterize, noise);
        d.X.vals[i] = sized.data;

        fill_truth_iseg(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip_h, flip_v, w/div, h/div);

        free_image(orig);
    }
    free(random_paths);
    return d;
}

data load_data_mask(int n, char **paths, int m, int w, int h, int classes, int boxes, int coords, int min, int max, float angle, float aspect, float hue, float saturation, float exposure, float hflip, float vflip, float solarize, float posterize, float noise)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, (coords+1)*boxes);

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        augment_args a = random_augment_args(orig, angle, aspect, min, max, w, h);
        image sized = rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);

        int flip_h = (rand() % 100) < (int)(hflip * 100.);
        int flip_v = (rand() % 100) < (int)(vflip * 100.);
        flip_image_x(sized, flip_h, flip_v);

        random_distort_image(sized, hue, saturation, exposure);
        random_distort_image_extend(sized, solarize, posterize, noise);
        d.X.vals[i] = sized.data;

        fill_truth_mask(random_paths[i], boxes, d.y.vals[i], classes, orig.w, orig.h, a, flip_h, flip_v, 14, 14);

        free_image(orig);
    }
    free(random_paths);
    return d;
}

data load_data_detection(int n, char **paths, char* label_dir, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure, float angle, float* zoom, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    d.y = make_matrix(n, 5*boxes);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        random_cutout_image(orig, cutout);

        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);

    	float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;
    	image rot = rotate_image_preserved(orig, rad);

        float dw = jitter * rot.w;
        float dh = jitter * rot.h;

        float new_ar = (rot.w + rand_uniform(-dw, dw)) / (rot.h + rand_uniform(-dh, dh));
        float scale = rand_uniform(zoom[0], zoom[1]); //nghiant: super important; inherit from old fixed_darknet framework

        float nw, nh;

        if(new_ar < 1){
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }

        float dx = rand_uniform(0, w - nw);
        float dy = rand_uniform(0, h - nh);

        place_image(rot, nw, nh, dx, dy, sized);

        random_distort_image(sized, hue, saturation, exposure);
        random_distort_image_extend(sized, solarize, posterize, noise);

        int flip_h = (rand() % 100) < (int)(hflip * 100.);
        int flip_v = (rand() % 100) < (int)(vflip * 100.);
        flip_image_x(sized, flip_h, flip_v);

        d.X.vals[i] = sized.data;

        fill_truth_detection(random_paths[i], label_dir, boxes, d.y.vals[i], classes, rad, orig.w, orig.h, rot.w, rot.h, flip_h, flip_v, -dx/w, -dy/h, nw/w, nh/h); //new method
     
        free_image(orig);
        free_image(rot);
    }
    free(random_paths);
    return d;
}

data load_data_auto_colorize(int n, char** paths, int m, int w, int h) {
    char** random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w;

    d.y = make_matrix(n, w*h*2);
    for (i = 0; i < n; ++i) {
        image orig = load_image_color(random_paths[i], 0, 0);
        image sized = resize_image(orig, w, h);
        rgb_to_lab(sized);
        d.X.vals[i] = sized.data;
        memcpy(d.y.vals[i], sized.data+w*h, 2*w*h*sizeof(float));
        // d.y.vals[i] = sized.data+w*h;
        free_image(orig);
    }
    free(random_paths);
    return d;
}

data load_data_letterbox_no_truth(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure, float angle, float* zoom, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);
        random_cutout_image(orig, cutout);

        image sized = make_image(w, h, orig.c);
        fill_image(sized, .5);

        float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;
        image rot = rotate_image_preserved(orig, rad);

        float dw = jitter * rot.w;
        float dh = jitter * rot.h;

        float new_ar = (rot.w + rand_uniform(-dw, dw)) / (rot.h + rand_uniform(-dh, dh));
        float scale = rand_uniform(zoom[0], zoom[1]); //nghiant: super important; inherit from old fixed_darknet framework

        float nw, nh;

        if(new_ar < 1){
            nh = scale * h;
            nw = nh * new_ar;
        } else {
            nw = scale * w;
            nh = nw / new_ar;
        }

        float dx = rand_uniform(0, w - nw);
        float dy = rand_uniform(0, h - nh);

        place_image(rot, nw, nh, dx, dy, sized);

        random_distort_image(sized, hue, saturation, exposure);
        random_distort_image_extend(sized, solarize, posterize, noise);

        int flip_h = (rand() % 100) < (int)(hflip * 100.);
        int flip_v = (rand() % 100) < (int)(vflip * 100.);
        flip_image_x(sized, flip_h, flip_v);

        d.X.vals[i] = sized.data;

        free_image(orig);
        free_image(rot);
    }
    free(random_paths);
    return d;
}

void *load_thread(void *ptr)
{
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == REGRESSION_DATA){
        *a.d = load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.hflip, a.vflip, a.solarize, a.posterize, a.noise, a.cutout);
    } else if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center, a.hflip, a.vflip, a.solarize, a.posterize, a.noise, a.cutout);
    } else if (a.type == ISEG_DATA){
        *a.d = load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.hflip, a.vflip, a.solarize, a.posterize, a.noise);
    } else if (a.type == INSTANCE_DATA){
        *a.d = load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.hflip, a.vflip, a.solarize, a.posterize, a.noise);
    } else if (a.type == SEGMENTATION_DATA){
        *a.d = load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.scale, a.hflip, a.vflip, a.solarize, a.posterize, a.noise);
    } else if (a.type == DETECTION_DATA){
        *a.d = load_data_detection(a.n, a.paths, a.label_dir, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure, a.angle, a.zoom, a.hflip, a.vflip, a.solarize, a.posterize, a.noise, a.cutout);
    } else if (a.type == LETTERBOX_DATA_NO_TRUTH){
        *a.d = load_data_letterbox_no_truth(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure, a.angle, a.zoom, a.hflip, a.vflip, a.solarize, a.posterize, a.noise, a.cutout);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    } else if (a.type == IMAGE_DATA_CROP){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = center_crop_image(*(a.im), a.w, a.h);
        fill_truth(a.path, a.labels, a.classes, a.truth);
    } else if (a.type == LETTERBOX_DATA){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    } else if (a.type == LETTERBOX_DATA_8BIT){
        *(a.im) = load_image_color(a.path, 0, 0);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
        double_im_to_255(*(a.resized));
    } else if (a.type == AUTO_COLORIZE_DATA){
        *a.d = load_data_auto_colorize(a.n, a.paths, a.m, a.w, a.h);
    } else if (a.type == LP_SUPERRES_DATA){
        *a.d = load_data_lp_superres(a.n, a.paths, a.m, a.w, a.h, a.us_factor);
    } else if (a.type == LP_SUPERRES_TF_DATA){
        *a.d = load_data_lp_superres_tf(a.n, a.all_data_im, a.all_data_gt, a.m, a.w, a.h, a.us_factor);
    } else if (a.type == LP_SUPERRES_TF_LESS_DATA){
        *a.d = load_data_lp_superres_tf_less(a.n, a.train_directory, a.im_prefix, a.gt_prefix, a.m, a.w, a.h, a.us_factor);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

void *load_threads(void *ptr)
{
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);
    data *buffers = calloc(args.threads, sizeof(data));
    pthread_t *threads = calloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i){
        args.thread_id = i;
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }
    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    free(threads);
    return 0;
}

pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args *ptr = calloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0);
    if(m) free(paths);
    return d;
}

data load_data_regression(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, 0, hflip, vflip, solarize, posterize, noise, cutout);
    d.y = load_regression_labels_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center, float hflip, float vflip, float solarize, float posterize, float noise, cutout_args cutout)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.w=size;
    d.h=size;
    d.X = load_image_augment_paths(paths, n, min, max, size, angle, aspect, hue, saturation, exposure, center, hflip, vflip, solarize, posterize, noise, cutout);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy);
    if(m) free(paths);
    return d;
}

data load_data_lp_superres(int n, char** paths, int m, int w, int h, int us_factor) {
    char** random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w;
    
    int gt_w = w * us_factor;
    int gt_h = h * us_factor;

    d.y = make_matrix(n, gt_w*gt_h);

    for (i = 0; i < n; ++i) {
        image orig = load_image_color(random_paths[i], 0, 0);
        image crop = random_crop_image(orig, gt_w, gt_h);
        image lowr = resize_image(crop, w, h);
        
        rgb_to_ycbcr(lowr);
        d.X.vals[i] = lowr.data;

        rgb_to_ycbcr(crop);
        memcpy(d.y.vals[i], crop.data, gt_w*gt_h*sizeof(float));
        free_image(orig);
        free_image(crop);
    }
    free(random_paths);
    return d;
}

int *get_random_data_index(int n, int m)
{
    int *indices = (int*)calloc(n, sizeof(int));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        indices[i] = rand() % m;
    }
    pthread_mutex_unlock(&mutex);
    return indices;
}


data load_data_lp_superres_tf_less(int n, char *train_directory, char *im_prefix, char *gt_prefix, int m, int w, int h, int us_factor) {
    int *random_indices = get_random_data_index(n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w;
    
    int gt_w = w * us_factor;
    int gt_h = h * us_factor;

    d.y.rows = n;
    d.y.vals = calloc(d.y.rows, sizeof(float *));
    d.y.cols = gt_w*gt_h;

    for (i = 0; i < n; ++i) {
        char file_name[256];
        FILE *fid;

        d.X.vals[i] = (float*)calloc(w * h, sizeof(float));
        sprintf(file_name, "%s%s_%d", train_directory, im_prefix, random_indices[i]);
        fid = fopen(file_name, "rb");
        fread((void*)(d.X.vals[i]), sizeof(float), w*h, fid);
        fclose(fid);

        d.y.vals[i] = (float*)calloc(gt_w * gt_h, sizeof(float));
        sprintf(file_name, "%s%s_%d", train_directory, gt_prefix, random_indices[i]);
        fid = fopen(file_name, "rb");
        fread((void*)(d.y.vals[i]), sizeof(float), gt_w * gt_h, fid);
        fclose(fid);

    }
    free(random_indices);
    return d;
}

data load_data_lp_superres_tf(int n, float** all_data_im, float** all_data_gt, int m, int w, int h, int us_factor) {
    int *random_indices = get_random_data_index(n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w;
    
    int gt_w = w * us_factor;
    int gt_h = h * us_factor;

    // d.y = make_matrix(n, gt_w*gt_h);

    d.y.rows = n;
    d.y.vals = calloc(d.y.rows, sizeof(float *));
    d.y.cols = gt_w*gt_h;

    for (i = 0; i < n; ++i) {
        d.X.vals[i] = all_data_im[random_indices[i]];
        d.y.vals[i] = all_data_gt[random_indices[i]];
        // memcpy(d.y.vals[i], all_data_gt[random_indices[i]], gt_w*gt_h*sizeof(float));
    }
    free(random_indices);
    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = calloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    d.w = d1.w;
    d.h = d1.h;
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data new = concat_data(d[i], out);
        free_data(out);
        out = new;
    }
    return out;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        if(y) memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data copy_data(data d)
{
    data c = {0};
    c.w = d.w;
    c.h = d.h;
    c.shallow = 0;
    c.num_boxes = d.num_boxes;
    c.boxes = d.boxes;
    c.X = copy_matrix(d.X);
    c.y = copy_matrix(d.y);
    return c;
}

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = calloc(num, sizeof(float *));
    r.y.vals = calloc(num, sizeof(float *));

    int i;
    for(i = 0; i < num; ++i){
        int index = rand()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}