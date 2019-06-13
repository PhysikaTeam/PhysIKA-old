#include "GLApp.h"
namespace Physika{
GLApp::GLApp(double time,double deltx,int width,int height){
	dt=time;
	dx=deltx;
	window_width=width;
	window_height=height;
}
GLApp::~GLApp(){}
void GLApp::init(std::string const surface,std::string const height,std::string const vx,std::string const vy){
	load_field_data(vx, initial_velocity_x, x_cells, y_cells);
	load_field_data(vy, initial_velocity_y, x_cells, y_cells);
	load_field_data(height, initial_height, x_cells, y_cells);
	load_field_data(surface, initial_surface_level, x_cells, y_cells);
	assert(x_cells * y_cells == initial_height.size());
	solver.set(x_cells,y_cells,dt,dx);
}
void GLApp::showframe(){
	VisualEngine visual_engine.set(x_cells, y_cells, dx, window_width, window_height);
	std::vector<GLfloat> water_height(initial_height.begin(), initial_height.end());
	std::vector<GLfloat> surface_level(initial_surface_level.begin(), initial_surface_level.end());
	for (size_t i = 0; i < water_height.size(); ++i) {
		water_height[i] += surface_level[i];
	}
	visual_engine.update_vertex_values(&water_height, &surface_level);
	while (1) {
	        visual_engine.render();
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
