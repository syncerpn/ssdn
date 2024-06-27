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

void accumulate_cpu(int N, int K, float *X, int INCX, float *Y, int INCY) {
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < K; ++j) {
			Y[i*INCY] += X[(i*K+j)*INCX];
		}
	}
}

void tile_repeat(int N, int K, int M, float *X, int INCX, float *Y, int INCY) {
	for (int i = 0; i < N / K; ++i) {
		for (int j = 0; j < M; ++j) {
			copy_cpu(K, X+i*K*INCX, INCX, Y+(i*K*M+j)*INCY, INCY);
		}
	}
}

void add_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
	if (Y == 0) {
		Y = X;
		INCY = INCX;
	}
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = X[i*INCX] + ALPHA;
	}
}

void scale_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
	if (Y == 0) {
		Y = X;
		INCY = INCX;
	}
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = X[i*INCX] * ALPHA;
	}
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
	if (Y == 0) {
		Y = X;
		INCY = INCX;
	}
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = pow(X[i*INCX], ALPHA);
	}
}

void constrain_cpu(int N, float MIN, float MAX, float *X, int INCX, float *Y, int INCY) {
	if (Y == 0) {
		Y = X;
		INCY = INCX;
	}
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = fminf(MAX, fmaxf(MIN, X[i*INCX]));
	}
}
