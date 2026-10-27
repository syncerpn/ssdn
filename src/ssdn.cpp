#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "ssdn.h"

extern int draft(int argc, char** argv);
extern int malm_idag_m3_2x(int argc, char** argv);
extern int malm_edsr_2x(int argc, char** argv);

int main(int argc, char** argv) {
	std::cout << "[INFO] ssdn run: ";
	for (int i = 0; i < argc; ++i) {
		std::cout << " " << argv[i];
		if (i == argc - 1) std::cout << std::endl;
	}

	if (argc == 1) {
		std::cout << "[ERRO] app is required" << std::endl;
		return 1;
	}

	if (0 == strcmp(argv[1], "draft")) return draft(argc, argv); // draft
	else if (0 == strcmp(argv[1], "malm_idag_m3_2x")) return malm_idag_m3_2x(argc, argv); // malm_idag_m3_2x
	else if (0 == strcmp(argv[1], "malm_edsr_2x")) return malm_edsr_2x(argc, argv); // malm_edsr_2x
	

	return 0;
}