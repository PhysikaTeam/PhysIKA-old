#include "Physika_Shallow_Water/Shallow_water_model/ShallowWaterSolver.h"
#include "Physika_Shallow_Water/Shallow_Water_Render/render.h"
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
void load_field_data(std::string const file_name, std::vector<double> &field, size_t &x_cells, size_t &y_cells);
void run(std::string const height,std::string const surface,std::string const vx,std::string const vy);
