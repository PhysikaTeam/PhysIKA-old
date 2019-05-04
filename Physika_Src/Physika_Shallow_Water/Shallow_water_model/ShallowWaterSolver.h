#include "Physika_Shallow_Water/Shallow_Water_meshs/Field.h"
namespace Physika {
class ShallowWaterSolver {
public:
	ShallowWaterSolver();
	ShallowWaterSolver(size_t x_cells, size_t y_cells, double time_step, double dx, double gravity = 9.8);
	void set(size_t x_cells, size_t y_cells, double time_step, double dx, double gravity = 9.8);
	void initialize_water_height(const std::vector<double> &input);
	void initialize_vx(const std::vector<double> &input);
	void initialize_vy(const std::vector<double> &input);
	void initialize_surface_level(const std::vector<double> &input);
	void run(size_t iterations);
	const Field &getWater_height() const;
	const Field &getSurface_level() const;
	double getTime_elapsed() const;
	void output(size_t iteration);
private:
	void euler();
	Field advect_height(double time_step) const;
	Field advect_vx(double time_step) const;
	Field advect_vy(double time_step) const;
	void update_height(double time_step);
	void update_vx(double time_step);
	void update_vy(double time_step)
	void apply_reflecting_boundary_conditions();
	double dt;
	double dx;
	double time_elapsed;
	size_t x_cells;
	size_t y_cells;
	Field water_height;
	Field vx;
	Field vy;
	Field surface_level;
	double const gravity;
};
}
