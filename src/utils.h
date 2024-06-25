#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "darknet.h"
#include "list.h"

#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

double what_time_is_it_now();
void free_ptrs(void **ptrs, int n);
void find_replace(char *str, char *orig, char *rep, char *output);
void malloc_error();
void file_error(char *s);
void strip(char *s);
void print_statistics(float *a, int n);
void k_means(float* data, int n, float* centroid, int c, int max_iters);
float *parse_fields(char *line, int n);
float constrain(float min, float max, float a);
float rand_scale(float s);

float dist_array(float *a, float *b, int n, int sub);
float sec(clock_t clocks);
char *fgetl(FILE *fp);
char *copy_string(char *s);
int constrain_int(int a, int min, int max);
int rand_int(int min, int max);

int count_fields(char *line);
#endif