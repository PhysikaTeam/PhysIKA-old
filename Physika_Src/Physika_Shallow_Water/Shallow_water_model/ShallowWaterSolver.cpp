#include "ShallowWaterSolver.h"
namespace Physika {
ShallowWaterSolver::ShallowWaterSolver(
	size_t x_cells,
	size_t y_cells,
	double time_step,
	double dx,
	double gravity) :

	dt(time_step),
	dx(dx),
	time_elapsed(0.0),
	x_cells(x_cells),
	y_cells(y_cells),
	water_height(x_cells, y_cells),
	vx(x_cells + 1, y_cells),
	vy(x_cells, y_cells + 1),
	surface_level(x_cells, y_cells),
	gravity(gravity) {}

void ShallowWaterSolver::run(size_t iterations) {

	for (size_t iteration = 0; iteration < iterations; ++iteration) {

		euler();
		apply_reflecting_boundary_conditions();
	}
}
Field ShallowWaterSolver::advect_height(double time_step) const {
	Field advected_height(water_height);
	for (size_t i = 1; i < x_cells - 1; ++i) {
		for (size_t j = 1; j < y_cells - 1; ++j) {
			double x = (i + 0.5) * dx;
			double y = (j + 0.5) * dx;
			double v_x = (vx(i, j) + vx(i + 1, j)) / 2;
			double v_y = (vy(i, j) + vy(i, j + 1)) / 2;
			x -= v_x * time_step;
			y -= v_y * time_step;
			double x_boundary = x_cells * dx;
			double y_boundary = y_cells * dx;
			if (x < 0) {
				x = -x;
			}
			if (x > x_boundary) {
				x = -x + 2 * x_boundary;
			}
			if (y < 0) {
				y = -y;
			}
			if (y > y_boundary) {
				y = -y + 2 * y_boundary;
			}
			double advected_i = x / dx - 0.5;
			double advected_j = y / dx - 0.5;
			size_t advected_i_floor = (size_t)advected_i;
			size_t advected_j_floor = (size_t)advected_j;
			double i_offset = advected_i - advected_i_floor;
			double j_offset = advected_j - advected_j_floor;
			advected_height(i, j) = water_height(advected_i_floor, advected_j_floor) * (1.0 - i_offset) * (1.0 - j_offset)
				+ water_height(advected_i_floor + 1, advected_j_floor) * i_offset * (1.0 - j_offset)
				+ water_height(advected_i_floor, advected_j_floor + 1) * (1.0 - i_offset) * j_offset
				+ water_height(advected_i_floor + 1, advected_j_floor + 1) * i_offset * j_offset;
		}
	}

	return advected_height;
}

Field ShallowWaterSolver::advect_vx(double time_step) const {

	Field advected_vx(vx);
	for (size_t i = 1; i < x_cells; ++i) {
		for (size_t j = 1; j < y_cells - 1; ++j) {

			double x = i * dx;
			double y = (j + 0.5) * dx;

			double v_x = vx(i, j);
			double v_y = (vy(i, j) + vy(i, j + 1) + vy(i - 1, j) + vy(i - 1, j + 1)) / 4;

			x -= v_x * time_step;
			y -= v_y * time_step;

			double x_boundary = x_cells * dx;
			double y_boundary = y_cells * dx;

			if (x < 0) {
				x = -x;
			}
			if (x > x_boundary) {
				x = -x + 2 * x_boundary;
			}
			if (y < 0) {
				y = -y;
			}
			if (y > y_boundary) {
				y = -y + 2 * y_boundary;
			}

			double advected_i = x / dx;
			double advected_j = y / dx - 0.5;

			size_t advected_i_floor = (size_t)advected_i;
			size_t advected_j_floor = (size_t)advected_j;

			double i_offset = advected_i - advected_i_floor;
			double j_offset = advected_j - advected_j_floor;

			assert((i_offset >= 0.0) && (j_offset >= 0.0) && (i_offset <= 1.0) && (j_offset <= 1.0));

			advected_vx(i, j) = vx(advected_i_floor, advected_j_floor) * (1.0 - i_offset) * (1.0 - j_offset)
				+ vx(advected_i_floor + 1, advected_j_floor) * i_offset * (1.0 - j_offset)
				+ vx(advected_i_floor, advected_j_floor + 1) * (1.0 - i_offset) * j_offset
				+ vx(advected_i_floor + 1, advected_j_floor + 1) * i_offset * j_offset;
		}
	}

	return advected_vx;
}

Field ShallowWaterSolver::advect_vy(double time_step) const {

	Field advected_vy(vy);
	for (size_t i = 1; i < x_cells - 1; ++i) {
		for (size_t j = 1; j < y_cells; ++j) {

			double x = (i + 0.5) * dx;
			double y = j * dx;

			double v_x = (vx(i, j) + vx(i + 1, j) + vx(i, j - 1) + vx(i + 1, j - 1)) / 4;
			double v_y = vy(i, j);

			x -= v_x * time_step;
			y -= v_y * time_step;

			double x_boundary = x_cells * dx;
			double y_boundary = y_cells * dx;

			if (x < 0) {
				x = -x;
			}
			if (x > x_boundary) {
				x = -x + 2 * x_boundary;
			}
			if (y < 0) {
				y = -y;
			}
			if (y > y_boundary) {
				y = -y + 2 * y_boundary;
			}

			double advected_i = x / dx - 0.5;
			double advected_j = y / dx;

			size_t advected_i_floor = (size_t)advected_i;
			size_t advected_j_floor = (size_t)advected_j;

			double i_offset = advected_i - advected_i_floor;
			double j_offset = advected_j - advected_j_floor;

			assert((i_offset >= 0.0) && (j_offset >= 0.0) && (i_offset <= 1.0) && (j_offset <= 1.0));

			advected_vy(i, j) = vy(advected_i_floor, advected_j_floor) * (1.0 - i_offset) * (1.0 - j_offset)
				+ vy(advected_i_floor + 1, advected_j_floor) * i_offset * (1.0 - j_offset)
				+ vy(advected_i_floor, advected_j_floor + 1) * (1.0 - i_offset) * j_offset
				+ vy(advected_i_floor + 1, advected_j_floor + 1) * i_offset * j_offset;
		}
	}

	return advected_vy;
}

void ShallowWaterSolver::update_height(double time_step) {

	for (size_t i = 1; i < x_cells - 1; ++i) {
		for (size_t j = 1; j < y_cells - 1; ++j) {

			double divergence = (vx(i + 1, j) - vx(i, j)
				+ vy(i, j + 1) - vy(i, j)) / dx;
			water_height(i, j) -= water_height(i, j) * divergence * time_step;
		}
	}
}

void ShallowWaterSolver::update_vx(double time_step) {

	for (size_t j = 1; j < y_cells - 1; ++j) {
		for (size_t i = 2; i < x_cells - 1; ++i) {

			double height_above_zero_left = surface_level(i - 1, j) + water_height(i - 1, j);
			double height_above_zero_right = surface_level(i, j) + water_height(i, j);

			vx(i, j) += gravity * (height_above_zero_left - height_above_zero_right) / dx * time_step;
		}
	}
}

void ShallowWaterSolver::update_vy(double time_step) {
	for (size_t j = 2; j < y_cells - 1; ++j) {
		for (size_t i = 1; i < x_cells - 1; ++i) {

			double height_above_zero_down = surface_level(i, j - 1) + water_height(i, j - 1);
			double height_above_zero_up = surface_level(i, j) + water_height(i, j);

			vy(i, j) += gravity * (height_above_zero_down - height_above_zero_up) / dx * time_step;
		}
	}
}

void ShallowWaterSolver::apply_reflecting_boundary_conditions() {

	for (size_t j = 0; j < y_cells; ++j) {
		water_height(0, j) = water_height(1, j) + surface_level(1, j) - surface_level(0, j);

		vx(1, j) = 0.0;
		vy(0, j) = 0.0;
	}

	for (size_t i = 0; i < x_cells; ++i) {
		water_height(i, 0) = water_height(i, 1);

		vy(i, 1) = 0.0;
		vx(i, 0) = 0.0;
	}

	for (size_t j = 0; j < y_cells; ++j) {
		water_height(x_cells - 1, j) = water_height(x_cells - 2, j) + surface_level(x_cells - 2, j) - surface_level(x_cells - 1, j);

		vx(x_cells - 1, j) = 0.0;
		vy(x_cells - 1, j) = 0.0;
	}

	for (size_t i = 0; i < x_cells; ++i) {
		water_height(i, y_cells - 1) = water_height(i, y_cells - 2) + surface_level(i, y_cells - 2) - surface_level(i, y_cells - 1);

		vy(i, y_cells - 1) = 0.0;
		vx(i, y_cells - 1) = 0.0;
	}
}
void ShallowWaterSolver::initialize_water_height(const std::vector<double> &input) {
	for (size_t i = 0; i < x_cells; ++i) {
		for (size_t j = 0; j < y_cells; ++j) {
			water_height(i, j) = input[i * x_cells + j];
		}
	}
}

void ShallowWaterSolver::initialize_vx(const std::vector<double> &input) {
	for (size_t i = 0; i <= x_cells; ++i) {
		for (size_t j = 0; j < y_cells; ++j) {
			vx(i, j) = input[i * x_cells + j];
		}
	}
}

void ShallowWaterSolver::initialize_vy(const std::vector<double> &input) {
	for (size_t i = 0; i < x_cells; ++i) {
		for (size_t j = 0; j <= y_cells; ++j) {
			vy(i, j) = input[i * x_cells + j];
		}
	}
}

void ShallowWaterSolver::initialize_surface_level(const std::vector<double> &input) {
	for (size_t i = 0; i < x_cells; ++i) {
		for (size_t j = 0; j < y_cells; ++j) {
			surface_level(i, j) = input[i * x_cells + j];
		}
	}
}
double ShallowWaterSolver::getTime_elapsed() const {
	return time_elapsed;
}

const Field &ShallowWaterSolver::getWater_height() const {
	return water_height;
}

const Field &ShallowWaterSolver::getSurface_level() const {
	return surface_level;
}

void ShallowWaterSolver::output(size_t iteration) {
	std::stringstream ss;
	ss << "output/height" << iteration << ".txt";
	std::ofstream fout(ss.str());
	for (size_t x = 0; x < x_cells; ++x) {
		for (size_t y = 0; y < y_cells; ++y) {
			fout << water_height(x, y) << ' ';
		}
		fout << '\n';
	}
}

void ShallowWaterSolver::euler() {
	Field new_height(advect_height(dt));
	Field new_vx(advect_vx(dt));
	Field new_vy(advect_vy(dt));

	water_height = new_height; 
	vx = new_vx;
	vy = new_vy;

	update_height(dt);
	update_vx(dt);
	update_vy(dt);

	time_elapsed += dt;
}
}
