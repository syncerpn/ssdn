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

void tile_repeat_cpu(int N, int K, int M, float *X, int INCX, float *Y, int INCY) {
	for (int i = 0; i < N / K; ++i) {
		for (int j = 0; j < M; ++j) {
			copy_cpu(K, X+i*K*INCX, INCX, Y+(i*K*M+j*K)*INCY, INCY);
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

void min_cpu(int N, float MIN, float *X, int INCX, float *Y, int INCY) {
	if (Y == 0) {
		Y = X;
		INCY = INCX;
	}
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = fmaxf(MIN, X[i*INCX]);
	}
}

void max_cpu(int N, float MAX, float *X, int INCX, float *Y, int INCY) {
	if (Y == 0) {
		Y = X;
		INCY = INCX;
	}
	for (int i = 0; i < N; ++i) {
		Y[i*INCY] = fminf(MAX, X[i*INCX]);
	}
}

void padding_cpu(float* X, int w, int h, int c, int p, float* Y) {
	int pw = w + 2 * p;
	int ph = h + 2 * p;
	for (int ci = 0; ci < c; ++ci) {
		for (int i = p; i < ph - p; ++i) {
			copy_gpu(w, X+ci*h*w+(i-p)*w, 1, Y+ci*ph*pw+i*pw+p, 1);
		}
	}
}

void unrolling_cpu(float* X, int w, int h, int c, int k, int s, float* Y) {
	int yw = (w - k) / s + 1;
	int yh = (h - k) / s + 1;

	int y_size = yw * yh;
	int f_size = k * k * c;

	for (int hi = 0; hi < yh; ++hi) {
		for (int wi = 0; wi < yw; ++wi) {
			for (int ci = 0; ci < c; ++ci) {
				for (int ki = 0; ki < k; ++ki) {
					copy_gpu(k, X+ci*h*w+(hi*s+ki)*w+wi*s, 1, Y+(hi*yw+wi)*f_size+ci*k*k+ki*k, 1);
				}
			}
		}
	}
}

void flatten_arrange_cpu(float* X, float* Z, int w, int h, int s, float* Y) {
	for (int ni = 0; ni < s*s; ++ni) {
		int si = ni / s;
		int sj = ni % s;
		for (int hi = 0; hi < h; ++hi) {
			for (int wi = 0; wi < w; ++wi) {
				Y[(hi*s+si)*w*s+wi*s+sj] = X[hi*w+wi] + Z[(si*s+sj)*h*w+hi*w+wi];
			}
		}
	}
}