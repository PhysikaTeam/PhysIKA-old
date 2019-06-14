#include "SurfaceShow.h"
SurfaceShow::SurfaceShow(size_t x_cells, size_t z_cells,  float dx, int window_width, int window_height) {
	this->x_cells = x_cells;
	this->z_cells = z_cells;
	this->dx = dx;
	this->window_width = window_width;
	this->window_height = window_height;

}
SurfaceShow::~SurfaceShow() {

}
void SurfaceShow::set_constants(bool m_have_tensor, float m_fric_coef, float m_gamma, float m_dt, float g) {
	sim.set_initial_constants(m_have_tensor, m_fric_coef, m_gamma, m_dt, g);
}
void SurfaceShow::init(std::string const surface, std::string const height, std::string const vx, std::string const vy, std::string const vz, int situation = 3, int times = 0) {
	load_txt_io2(surface, initial_surface_level);
	load_txt_io2(height, initial_height);
	load_txt_io2(vx, initial_velocity_x);
	load_txt_io2(vy, initial_velocity_y);
	load_txt_io2(vz, initial_velocity_z);
	int i = 0;
	while (i < initial_velocity_x.size()) {
		initial_velocity.push_back(initial_velocity_x[i]);
		initial_velocity.push_back(initial_velocity_y[i]);
		initial_velocity.push_back(initial_velocity_z[i]);
	}
	sim.generate_origin(x_cells, z_cells, initial_surface_level, dx);
	sim.generate_mesh(situation, times);
	sim.init();
	std::vector<std::vector<float>> v;
	for (int j = 0,t=0; j < i; j++) {
		v[j].push_back(initial_velocity[t++]);
		v[j].push_back(initial_velocity[t++]);
		v[j].push_back(initial_velocity[t++]);
	}
	sim.set_initial_conditions(initial_height, v);
}
void SurfaceShow::showframe(int n) {
	VisualEngine visual_engine(x_cells, z_cells, dx, window_width, window_height);
	std::vector<GLfloat> water_height(sim.height.begin(), sim.height.end());
	std::vector<GLfloat> surface_level(sim.bottom.begin(), sim.bottom.end());
	visual_engine.update_vertex_values(&water_height, &surface_level);
	int i = 0;
	while (!visual_engine.should_stop()) {
		visual_engine.render();
		sim.runoneframe();
		if (true && i % 10 == 0) {
			water_height = sim.height;
			visual_engine.update_vertex_values(&water_height);
		}
		i++;
	}
	sim.clear();
}