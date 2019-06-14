#include "load_txt_io2.h"
void load_txt_io2(const std::string filename, std::vector<float> v) {
	std::ifstream input(filename);
	input.exceptions(std::istream::failbit | std::istream::badbit);
	for (float &value : v) {
		input >> value;
	}
}