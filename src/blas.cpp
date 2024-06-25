#include "blas.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fill_cpu(int N, float ALPHA, float *X, int INCX) {
	for (int i = 0; i < N; ++i) {
		X[i*INCX] = ALPHA;
	}
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] += ALPHA * X[i*INCX];
	}
}

void add_cpu(int N, float ALPHA, float *X, int INCX) {
	for (int i = 0; i < N; ++i) {
		X[i*INCX] += ALPHA;
	}
}

void scale_cpu(int N, float ALPHA, float *X, int INCX) {
	for (int i = 0; i < N; ++i) {
		X[i*INCX] *= ALPHA;
	}
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY) {
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] *= X[i*INCX];
	}
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY) {
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = X[i*INCX];
	}
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = pow(X[i*INCX], ALPHA);
	}
}

void constrain_cpu(int N, float MIN, float MAX, float *X, int INCX) {
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = fminf(MAX, fmaxf(MIN, X[i*INCX]));
	}
}
