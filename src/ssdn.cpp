#include <iostream>

#include "ssdn.h"

void lead_one_encode(int x, int &k, int &p) {
	if (x == 0 || x == 1) {
		k = 0;
		p = 0;
	}
	k = 0;
	int xx = x;
	while (xx > 1) {
		xx = xx >> 1;
		k += 1;
	}
	p = x - (1 << k);
}

void approximate(int i, int j) {
	if (i == 0 || j == 0) return 0;

	int sign_i = i > 0 ? 1 : -1;
	int sign_j = j > 0 ? 1 : -1;

	i = sign_i * i;
	j = sign_j * j;

	int ki, pi, kj, pj;
	lead_one_encode(i, ki, pi);
	lead_one_encode(j, kj, pj);

	int p1 = pi * pj;
	int p2 = ((1 << ki) - pi) * ((1 << kj) - pj);
	int e = p1 < p2 ? p1 : p2;
	return sign_i * sign_j * (i * j - e);
}

void run_sim_fast_approx_ma() {

}

int main() {
	int k, p;
	for (int i = 0; i <= 512; ++i) {
		for (int j = i; j <= 512; ++j) {
			std::cout << i << " x " << j << " = " << approximate(i, j) << std::endl;
		}
	}
	return 0;
}