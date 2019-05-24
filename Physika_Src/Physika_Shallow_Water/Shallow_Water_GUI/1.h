#pragma once
#include "simulator.h"
#include "render.h"
#include "input_txt_io2.h"
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
class SurfaceShow {
private:
	size_t x_cells;
	size_t z_cells;
	float dt;
	float dx;
	int window_width;
	int window_height;
	Simulator sim;
	std::vector<double> initial_height;
	std::vector<double> initial_velocity_x;
	std::vector<double> initial_velocity_y;
	std::vector<double> initial_velocity_z;
	std::vector<double> initial_surface_level;
	std::vector<double> initial_velocity;
	SurfaceShow(size_t x_cells, size_t z_cells, float dt, float dx, int window_width, int window_height);
	~SurfaceShow();
	void init(std::string const surface, std::string const height, std::string const vx, std::string const vy, std::string const vz);
	void showframe();
	void drawoneframe();
};
