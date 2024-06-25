#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include "utils.h"

unsigned int seed = 0;

double what_time_is_it_now() {
    struct timeval time;
    if (gettimeofday(&time,NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void error(const char *s) {
    perror(s);
    assert(0);
    exit(-1);
}

void malloc_error() {
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

void file_error(char *s) {
    fprintf(stderr, "Couldn't open file: \033[0;31m%s\033[0m\n", s);
    exit(0);
}