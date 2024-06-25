#ifndef INITIALIZER_H
#define INITIALIZER_H
#include "darknet.h"

void initialize_array(float* x, int nx, initializer init);
INITIALIZER get_initializer_type(char* type);
char* get_initializer_string(INITIALIZER type);

#endif