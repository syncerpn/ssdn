#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "logistic";
        case LOGGY:
            return "loggy";
        case RELU:
            return "relu";
        case ELU:
            return "elu";
        case SELU:
            return "selu";
        case RELIE:
            return "relie";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        case PLSE:
            return "plse";
        case LEAKY:
            return "leaky";
        case STAIR:
            return "stair";
        case HARDTAN:
            return "hardtan";
        case LHTAN:
            return "lhtan";
        case ABLEAKY:
            return "ableaky";
        case SWISH:
            return "swish";
        case MISH:
            return "mish";
        default:
            break;
    }
    return "relu";
}

char *get_activation_string_cap(ACTIVATION a)
{
    switch(a){
        case LOGISTIC:
            return "LOGISTIC";
        case LOGGY:
            return "LOGGY";
        case RELU:
            return "RELU";
        case ELU:
            return "ELU";
        case SELU:
            return "SELU";
        case RELIE:
            return "RELIE";
        case RAMP:
            return "RAMP";
        case LINEAR:
            return "LINEAR";
        case TANH:
            return "TANH";
        case PLSE:
            return "PLSE";
        case LEAKY:
            return "LEAKY";
        case STAIR:
            return "STAIR";
        case HARDTAN:
            return "HARDTAN";
        case LHTAN:
            return "LHTAN";
        case ABLEAKY:
            return "ABLEAKY";
        case SWISH:
            return "SWISH";
        case MISH:
            return "MISH";
        default:
            break;
    }
    return "RELU";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "logistic")==0) return LOGISTIC;
    if (strcmp(s, "loggy")==0) return LOGGY;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "relie")==0) return RELIE;
    if (strcmp(s, "plse")==0) return PLSE;
    if (strcmp(s, "hardtan")==0) return HARDTAN;
    if (strcmp(s, "lhtan")==0) return LHTAN;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "leaky")==0) return LEAKY;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "stair")==0) return STAIR;
    if (strcmp(s, "ableaky")==0) return ABLEAKY;
    if (strcmp(s, "swish")==0) return SWISH;
    if (strcmp(s, "mish")==0) return MISH;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

void print_activation_summary(activation_scheme act) {
    char buff[64];
    int arg_2 = 5;
    int arg_2d= 3;
    int arg_3 = 5;
    int arg_3d= 3;
    switch(act.type){
        case ABLEAKY:
            sprintf(buff, "%s %*.*f %*.*f", get_activation_string_cap(act.type), arg_2, arg_2d, act.alpha, arg_3, arg_3d, act.beta);
            break;
        case LOGISTIC:
        case LOGGY:
        case RELU:
        case ELU:
        case SELU:
        case RELIE:
        case RAMP:
        case LINEAR:
        case TANH:
        case PLSE:
        case LEAKY:
        case STAIR:
        case HARDTAN:
        case LHTAN:
        case SWISH:
        case MISH:
            sprintf(buff, "%s %*s- %*s-", get_activation_string_cap(act.type), arg_2-1, "", arg_3-1, "");
        default:
            break;
    }
    fprintf(stderr, " %20s ", buff);
    return;
}