#include "loadinputfile.h"
void load_field_data(std::string const file_name, std::vector<double> &field, size_t &x_cells, size_t &y_cells) {
	std::ifstream input(file_name);
	input.exceptions(std::istream::failbit | std::istream::badbit);

	input >> x_cells >> y_cells;

	field.resize(x_cells * y_cells);
	for (double &value : field) {
		input >> value;
	}
}
void run(std::string const height,std::string const surface,std::string const vx,std::string const vy){
        size_t x_cells;
	size_t y_cells;

	std::vector<double> initial_height;
	std::vector<double> initial_velocity_x;
	std::vector<double> initial_velocity_y;
	std::vector<double> initial_surface_level;

	load_field_data(vx, initial_velocity_x, x_cells, y_cells);
	load_field_data(vy, initial_velocity_y, x_cells, y_cells);
	load_field_data(height, initial_height, x_cells, y_cells);
	load_field_data(surface, initial_surface_level, x_cells, y_cells);
	assert(x_cells * y_cells == initial_height.size());
	double const time_step = 0.01;
	double const dx = 0.1;

	ShallowWaterSolver solver(x_cells, y_cells, time_step, dx);

	solver.initialize_water_height(initial_height);
	solver.initialize_vx(initial_velocity_x);
	solver.initialize_vy(initial_velocity_y);
	solver.initialize_surface_level(initial_surface_level);

	std::vector<GLfloat> water_height(initial_height.begin(), initial_height.end());
	std::vector<GLfloat> surface_level(initial_surface_level.begin(), initial_surface_level.end());
	for (size_t i = 0; i < water_height.size(); ++i) {
		water_height[i] += surface_level[i];
	}

	VisualEngine visual_engine(x_cells, y_cells, dx, 1200, 900);
	visual_engine.update_vertex_values(&water_height, &surface_level);
	while (!visual_engine.should_stop()) {
		visual_engine.render();
		if (true) {
			solver.run(1);
			std::vector<double> const *wh = &solver.getWater_height().getBase();
			water_height = std::vector<GLfloat>(wh->begin(), wh->end());
			for (size_t i = 0; i < water_height.size(); ++i) {
				water_height[i] += surface_level[i];
			}
			visual_engine.update_vertex_values(&water_height);
		}
	}

}
