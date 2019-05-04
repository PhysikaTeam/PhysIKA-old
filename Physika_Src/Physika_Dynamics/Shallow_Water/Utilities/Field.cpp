#include "Field.h"
namespace Physika{
Field::Field(size_t x_cells, size_t y_cells) :
	base(x_cells * y_cells, 0),
	x_cells(x_cells),
	y_cells(y_cells) {}

void Field::resize(size_t x_cells, size_t y_cells) {

	this->x_cells = x_cells;
	this->y_cells = y_cells;

	base.resize(x_cells * y_cells);
}

Scalar& Field::operator()(size_t x, size_t y) {

	assert((x < x_cells) && (y < y_cells));
	return base[x * y_cells + y];
}

Scalar Field::operator()(size_t x, size_t y) const {

	assert((x < x_cells) && (y < y_cells));
	return base[x * y_cells + y];
}

Field Field::operator+(Field const &field) const {

	Field sum(*this);
	sum += field;

	return sum;
}

Field Field::operator-(Field const &field) const {

	Field diff(*this);
	diff -= field;

	return diff;
}

Field& Field::operator+=(Field const &field) {
	assert(x_cells == field.x_cells && y_cells == field.y_cells);

	size_t size = field.base.size();
	for (size_t i = 0; i < size; ++i) {
		base[i] += field.base[i];
	}

	return *this;
}

Field &Field::operator-=(Field const &field) {
	assert(x_cells == field.x_cells && y_cells == field.y_cells);

	size_t size = field.base.size();
	for (size_t i = 0; i < size; ++i) {
		base[i] -= field.base[i];
	}

	return *this;
}

Field Field::operator*(double multiplier) const {

	Field scaled(*this);
	for (double &element : scaled.base) {
		element *= multiplier;
	}

	return scaled;
}

const std::vector<Scalar> &Field::getBase() const {
	return base;
}
}
