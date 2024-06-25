#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <time.h>
#include "ssdn.h"

#define TIME(a) \
    do { \
    double start = what_time_is_it_now(); \
    a; \
    printf("%s took: %f seconds\n", #a, what_time_is_it_now() - start); \
    } while (0)

#define TWO_PI 6.2831853071795864769252866f

double what_time_is_it_now();
void error(const char *s);
void malloc_error();
void file_error(char *s);
#endif