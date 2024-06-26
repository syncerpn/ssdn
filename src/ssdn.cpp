#include <iostream>

#include "ssdn.h"

void lead_one_encode(int x, int &k, int &p) {
	if (x == 0 || x == 1) {
		k = 0;
		p = 0;
	}
	k = 0;
	while (x > 1) {
		x = x >> 1;
		k += 1;
	}
	p = x - (1 << k);	
}

void run_sim_fast_approx_ma() {

}

int main() {
	int k, p;
	for (int i = 0; i <= 512; ++i) {
		lead_one_encode(i, k, p);
		std::cout << i << " " << k << " " << p << std::endl;
	}
	return 0;
}