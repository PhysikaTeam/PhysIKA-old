#pragma once
#include "simulator.h"
#include "render.h"
#include "load_txt_io2.h"
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
class SurfaceShow {
private:
	size_t x_cells;
	size_t z_cells;
	float dx;
	int window_width;
	int window_height;
	Simulator sim;
	std::vector<float> initial_height;
	std::vector<float> initial_velocity_x;
	std::vector<float> initial_velocity_y;
	std::vector<float> initial_velocity_z;
	std::vector<float> initial_surface_level;
	std::vector<float> initial_velocity;
	SurfaceShow(size_t x_cells, size_t z_cells, float dx, int window_width, int window_height);
	~SurfaceShow();
	void set_constants(bool m_have_tensor, float m_fric_coef, float m_gamma, float m_dt, float g);
	void init(std::string const surface, std::string const height, std::string const vx, std::string const vy, std::string const vz,int situation=3,int times=0);
	void showframe(int n);
};
	