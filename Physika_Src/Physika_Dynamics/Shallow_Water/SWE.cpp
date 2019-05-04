#include <cstdlib>
#include <limits>
#include <iostream>
#include <algorithm>
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Numerics/Linear_System_Solvers/conjugate_gradient_solver.h"
#include "Physika_Dynamics/Driver/driver_plugin_base.h"
#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"
#include "Physika_Dynamics/Utilities/Grid_Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
#include "Physika_Dynamics/Shallow_Water/SWE_plugin/swe_plugin_base.h"
#include "Physika_Dynamics/Shallow_Water/SWE.h"
namespace Physika{
template <typename Scalar, int Dim>
SWE<Scalar,Dim>::SWE(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                     const Grid<Scalar,Dim> &grid):grid_(grid){}
void SWE<Scalar,Dim>::addPlugin(DriverPluginBase<Scalar> *plugin)
{
    if(plugin==NULL)
    {
        std::cerr<<"Warning: NULL plugin provided, operation ignored!\n";
        return;
    }

    if(dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(plugin)==NULL)
    {
        std::cerr<<"Warning: Wrong type of plugin provided, operation ignored!\n";
        return;
    }
    plugin->setDriver(this);
    this->plugins_.push_back(plugin);
}
bool SWE<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::write(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::read(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>	
const Grid<Scalar,Dim>& SWE<Scalar,Dim>::grid() const
{
    return grid_;
}		 
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setGrid(const Grid<Scalar,Dim> &grid)
{
    grid_ = grid;
}
template <typename Scalar, int Dim>
Field<Scalar> SWE<Scalar,Dim>::advect_height(int timestep){
	Field advected_height(grid_height_);
	for (size_t i = 1; i < x_cells - 1; ++i) {
		for (size_t j = 1; j < y_cells - 1; ++j) {
			Scalar x = (i + 0.5) * dx;
			Scalar y = (j + 0.5) * dx;
			Scalar v_x = (vx(i, j) + vx(i + 1, j)) / 2;
			Scalar v_y = (vy(i, j) + vy(i, j + 1)) / 2;
			x -= v_x * time_step;
			y -= v_y * time_step;
			Scalar x_boundary = x_cells * dx;
			Scalar y_boundary = y_cells * dx;
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
			Scalar advected_i = x / dx - 0.5;
			Scalar advected_j = y / dx - 0.5;
			size_t advected_i_floor = (size_t)advected_i;
			size_t advected_j_floor = (size_t)advected_j;
			Scalar i_offset = advected_i - advected_i_floor;
			Scalar j_offset = advected_j - advected_j_floor;
			advected_height(i, j) = water_height(advected_i_floor, advected_j_floor) * (1.0 - i_offset) * (1.0 - j_offset)
				+ water_height(advected_i_floor + 1, advected_j_floor) * i_offset * (1.0 - j_offset)
				+ water_height(advected_i_floor, advected_j_floor + 1) * (1.0 - i_offset) * j_offset
				+ water_height(advected_i_floor + 1, advected_j_floor + 1) * i_offset * j_offset;
		}
	}

	return advected_height;
}
template <typename Scalar, int Dim>
Field<Scalar> SWE<Scalar,Dim>::advect_vx(Scalar timestep){
	Field advected_vx(grid_vx_);
	for (size_t i = 1; i < x_cells; ++i) {
		for (size_t j = 1; j < y_cells - 1; ++j) {

			Scalar x = i * dx;
			Scalar y = (j + 0.5) * dx;

			Scalar v_x = vx(i, j);
			Scalar v_y = (vy(i, j) + vy(i, j + 1) + vy(i - 1, j) + vy(i - 1, j + 1)) / 4;

			x -= v_x * time_step;
			y -= v_y * time_step;

			Scalar x_boundary = x_cells * dx;
			Scalar y_boundary = y_cells * dx;

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

			Scalar advected_i = x / dx;
			Scalar advected_j = y / dx - 0.5;

			size_t advected_i_floor = (size_t)advected_i;
			size_t advected_j_floor = (size_t)advected_j;

			Scalar i_offset = advected_i - advected_i_floor;
			Scalar j_offset = advected_j - advected_j_floor;

			assert((i_offset >= 0.0) && (j_offset >= 0.0) && (i_offset <= 1.0) && (j_offset <= 1.0));

			advected_vx(i, j) = vx(advected_i_floor, advected_j_floor) * (1.0 - i_offset) * (1.0 - j_offset)
				+ vx(advected_i_floor + 1, advected_j_floor) * i_offset * (1.0 - j_offset)
				+ vx(advected_i_floor, advected_j_floor + 1) * (1.0 - i_offset) * j_offset
				+ vx(advected_i_floor + 1, advected_j_floor + 1) * i_offset * j_offset;
		}
	}

	return advected_vx;
}
template <typename Scalar, int Dim>
Field<Scalar> SWE<Scalar,Dim>::advect_vy(Scalar timestep){
	Field advected_vy(grid_vy_);
	for (size_t i = 1; i < x_cells - 1; ++i) {
		for (size_t j = 1; j < y_cells; ++j) {

			Scalar x = (i + 0.5) * dx;
			Scalar y = j * dx;

			Scalar v_x = (vx(i, j) + vx(i + 1, j) + vx(i, j - 1) + vx(i + 1, j - 1)) / 4;
			Scalar v_y = vy(i, j);

			x -= v_x * time_step;
			y -= v_y * time_step;

			Scalar x_boundary = x_cells * dx;
			Scalar y_boundary = y_cells * dx;

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

			Scalar advected_i = x / dx - 0.5;
			Scalar advected_j = y / dx;

			size_t advected_i_floor = (size_t)advected_i;
			size_t advected_j_floor = (size_t)advected_j;

			Scalar i_offset = advected_i - advected_i_floor;
			Scalar j_offset = advected_j - advected_j_floor;

			assert((i_offset >= 0.0) && (j_offset >= 0.0) && (i_offset <= 1.0) && (j_offset <= 1.0));

			advected_vy(i, j) = vy(advected_i_floor, advected_j_floor) * (1.0 - i_offset) * (1.0 - j_offset)
				+ vy(advected_i_floor + 1, advected_j_floor) * i_offset * (1.0 - j_offset)
				+ vy(advected_i_floor, advected_j_floor + 1) * (1.0 - i_offset) * j_offset
				+ vy(advected_i_floor + 1, advected_j_floor + 1) * i_offset * j_offset;
		}
	}

	return advected_vy;
}
template <typename Scalar, int Dim>
void  SWE<Scalar,Dim>::update_height(Scalar timestep){
	for (size_t i = 1; i < x_cells - 1; ++i) {
		for (size_t j = 1; j < y_cells - 1; ++j) {

			Scalar divergence = (vx(i + 1, j) - vx(i, j)
				+ vy(i, j + 1) - vy(i, j)) / dx;
			water_height(i, j) -= water_height(i, j) * divergence * time_step;
		}
	}
}
template <typename Scalar, int Dim>
void  SWE<Scalar,Dim>::update_vx(Scalar timestep){
	for (size_t j = 1; j < y_cells - 1; ++j) {
		for (size_t i = 2; i < x_cells - 1; ++i) {

			Scalar height_above_zero_left = surface_level(i - 1, j) + water_height(i - 1, j);
			Scalar height_above_zero_right = surface_level(i, j) + water_height(i, j);

			vx(i, j) += gravity * (height_above_zero_left - height_above_zero_right) / dx * time_step;
		}
	}
}
template <typename Scalar, int Dim>
void  SWE<Scalar,Dim>::update_vy(Scalar timestep){
	for (size_t j = 2; j < y_cells - 1; ++j) {
		for (size_t i = 1; i < x_cells - 1; ++i) {

			Scalar height_above_zero_down = grid_surface_(i, j - 1) + grid_height_(i, j - 1);
			Scalar height_above_zero_up = grid_surface_(i, j) + grid_height_(i, j);

			vy(i, j) += gravity * (height_above_zero_down - height_above_zero_up) / dx * time_step;
		}
	}
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setgravity(const Scalar &g){
	gravity=g;
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setdx(const Scalar &x){
	dx=x;
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::elur(Scalar dt){
  Field new_height(advect_height(dt));
	Field new_vx(advect_vx(dt));
	Field new_vy(advect_vy(dt));

	grid_height_ = new_height; 
	grid_vx_ = new_vx;
	grid_vy_ = new_vy;

	update_height(dt);
	update_vx(dt);
	update_vy(dt);
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setboundarycondition(){
       for (size_t j = 0; j < y_cells; ++j) {
		grid_height_(0, j) = grid_height_(1, j) + grid_surface_(1, j) - grid_surface_(0, j);

		grid_vx_(1, j) = 0.0;
		grid_vy_(0, j) = 0.0;
	}

	for (size_t i = 0; i < x_cells; ++i) {
		grid_height_(i, 0) = grid_height_(i, 1);

		grid_vy_(i, 1) = 0.0;
		grid_vx_(i, 0) = 0.0;
	}

	for (size_t j = 0; j < y_cells; ++j) {
		grid_height_(x_cells - 1, j) = grid_height_(x_cells - 2, j) + grid_surface_(x_cells - 2, j) - grid_surface_(x_cells - 1, j);

		grid_vx_(x_cells - 1, j) = 0.0;
		grid_vy_(x_cells - 1, j) = 0.0;
	}

	for (size_t i = 0; i < x_cells; ++i) {
		grid_height_(i, y_cells - 1) = grid_height_(i, y_cells - 2) + grid_surface_(i, y_cells - 2) - grid_surface_(i, y_cells - 1);

		grid_vy_(i, y_cells - 1) = 0.0;
		grid_vx_(i, y_cells - 1) = 0.0;
	}
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setxy(const size_t &x,const size_t &y){
    x_cells=x;
    y_cells=y;
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setvx(const std::vector<Scalar> &input){
       for (size_t i = 0; i <= x_cells; ++i) {
		for (size_t j = 0; j < y_cells; ++j) {
			grid_vx_(i, j) = input[i * x_cells + j];
		}
	}
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setvy(const std::vector<Scalar> &input)
{
	for (size_t i = 0; i <= x_cells; ++i) {
		for (size_t j = 0; j < y_cells; ++j) {
			grid_vy_(i, j) = input[i * x_cells + j];
		}
	}
}
template <typename Scalar, int Dim>
void SWE<Scalar,Dim>::setsurface(const std::vector<Scalar> &input){
     for (size_t i = 0; i <= x_cells; ++i) {
		for (size_t j = 0; j < y_cells; ++j) {
			grid_surface_(i, j) = input[i * x_cells + j];
		}
	}
}
template <typename Scalar, int Dim>
      for (size_t i = 0; i <= x_cells; ++i) {
		for (size_t j = 0; j < y_cells; ++j) {
			grid_height_(i, j) = input[i * x_cells + j];
		}
	}
}
template class SWE<float,2>;
template class SWE<double,2>;
}
