#ifndef PHYSIKA_DYNAMICS_SHALLOW_WATER_UTILITIES_FIELD_H_
#define PHYSIKA_DYNAMICS_SHALLOW_WATER_UTILITIES_FIELD_H_
#include <vector>
#include <stddef.h>
#include <omp.h>
namespace Physika{
template <typename Scalar>
class Field {
public:

	Field(size_t x_cells = 0, size_t y_cells = 0);
	Field(Field const & copy_from) = default;
	Field(Field&& move_from) = default;

	Scalar &operator()(size_t x, size_t y);
	Scalar  operator()(size_t x, size_t y) const;

	void resize(size_t x_cells, size_t y_cells);

	Field& operator=(Field const &copy_from) = default;

	Field operator+(Field const &field) const;
	Field operator-(Field const &field) const;
	Field& operator+=(Field const &field);
	Field& operator-=(Field const &field);
	Field operator*(double multiplier) const;

	const std::vector<double> &getBase() const;

private:

	std::vector<Scalar> base;

	size_t x_cells;
	size_t y_cells;
};
}
#endif;
