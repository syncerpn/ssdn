#include <iostream>
#include <math.h>
#include <sys/stat.h>

#include <vector>
#include <unordered_map>
#include <string>
#include <thread> // For std::this_thread::sleep_for
#include <chrono> // For std::chrono::seconds

#include "ssdn.h"

static size_t N = 1;
static size_t C = 1;
static size_t H = 1;
static size_t W = 1;
static size_t K = 1;

static int parse(int argc, char** argv) {
	if (argc >= 3) {
		N = std::stoi(argv[2]);
	}
	if (argc >= 4) {
		C = std::stoi(argv[3]);
	}
	if (argc >= 5) {
		H = std::stoi(argv[4]);
	}
	if (argc >= 6) {
		W = std::stoi(argv[5]);
	}
	if (argc >= 7) {
		K = std::stoi(argv[6]);
	}
	return 0;
}

int draft(int argc, char** argv) {
	if (parse(argc, argv)) {
		std::cout << "[ERRO] parsing error" << std::endl;
		return 1;
	}

	size_t size = N * C * H * W * K;
	std::cout << "[INFO] " << "try allocating (" << N << " x " << C << " x " << H << " x " << W << " x " << K <<"): " << size << std::endl;
	float* x = cuda_make_array(0, size);

    std::this_thread::sleep_for(std::chrono::seconds(5));	

    cuda_free(x);

	return 0;
}