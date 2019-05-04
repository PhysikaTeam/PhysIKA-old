#include "input_txt_io.h"
namespace Physika {
void load_field_data(std::string const file_name, std::vector<double> &field, size_t &x_cells, size_t &y_cells) {
	std::ifstream input(file_name);
	input.exceptions(std::istream::failbit | std::istream::badbit);

	input >> x_cells >> y_cells;

	field.resize(x_cells * y_cells);
	for (double &value : field) {
		input >> value;
	}
}
}
