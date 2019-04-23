#include <vector>
#include <stddef.h>
#include <omp.h>


class Field {
public:

	Field(size_t x_cells = 0, size_t y_cells = 0);
	Field(Field const & copy_from) = default;
	Field(Field&& move_from) = default;

	double &operator()(size_t x, size_t y);
	double  operator()(size_t x, size_t y) const;

	void resize(size_t x_cells, size_t y_cells);

	Field& operator=(Field const &copy_from) = default;

	Field operator+(Field const &field) const;
	Field operator-(Field const &field) const;
	Field& operator+=(Field const &field);
	Field& operator-=(Field const &field);
	Field operator*(double multiplier) const;

	const std::vector<double> &getBase() const;

private:

	std::vector<double> base;

	size_t x_cells;
	size_t y_cells;
};
