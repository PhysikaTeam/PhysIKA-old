#pragma once

#include <math.h>

#include "forceline.h"
#include "vec3f.h"

#include <algorithm>
using namespace std;

#include <assert.h>
#include <string.h>

class matrix3f {
	double _data[9]; // stored in column-major format

	// private. Clients should use named constructors colMajor or rowMajor.
	matrix3f(const double data[9]) {
		//std::copy(data, data+9, _data);
		memcpy(_data, data, sizeof(double)*9);
	}

public:
     // ----- static member functions -----

   //! Named constructor: construct from double array, column-major storage
	static matrix3f colMajor( const double data[ 9 ] ) {
		return matrix3f(data);
	}

    //! Named constructor: construct from double array, row-major storage
	static matrix3f rowMajor( const double data[ 9 ] ) {
		return matrix3f(
			data[0], data[3], data[6],
			data[1], data[4], data[7],
			data[2], data[5], data[8]);
	}

    //! Named constructor: construct a scaling matrix, with [sx sy sz] along
    //! the main diagonal
	static matrix3f scaling( double sx, double sy, double sz ) {
		return matrix3f(
			sx, 0, 0,
			0, sy, 0,
			0, 0, sz);
	}

    //! Named constructor: construct a rotation matrix, representing a
    //! rotation of theta about the given axis.
	static matrix3f rotation( const vec3f& axis, double theta ) {
		const double s = sin( theta );
		const double c = cos( theta );
		const double t = 1-c;
		const double x = axis.x, y = axis.y, z = axis.z;
		return matrix3f(
			t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
			t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
			t*x*z - s*y, t*y*z + s*x, t*z*z + c
		);
	}

    //! Named constructor: create matrix M = a * b^T
	static matrix3f outerProduct( const vec3f& a, const vec3f& b ) {
		return matrix3f(
			a.x * b.x, a.x * b.y, a.x * b.z,
			a.y * b.x, a.y * b.y, a.y * b.z,
			a.z * b.x, a.z * b.y, a.z * b.z
		);
	}

    //! Named constructor: create identity matrix. This is implemented this
    //! way (instead of as a static member) so that the compiler knows the
    //! contents of the matrix and can optimise it out.
	static matrix3f identity() {
		static const double entries[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
		return matrix3f(entries);
	}

    //! Named constructor: create zero matrix
	static matrix3f zero() {
		static const double entries[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		return matrix3f(entries);
	}

    // ----- member functions -----
    
	matrix3f() {
		*this = matrix3f::zero();
	}

    matrix3f(double entry00, double entry01, double entry02,
			        double entry10, double entry11, double entry12,
					double entry20, double entry21, double entry22) {
		_data[0] = entry00, _data[3] = entry01, _data[6] = entry02;
		_data[1] = entry10, _data[4] = entry11, _data[7] = entry12;
		_data[2] = entry20, _data[5] = entry21, _data[8] = entry22;
	}

    // Default copy constructor is fine.
    
    // Default assignment operator is fine.
	double operator()( size_t row, size_t col ) const {
		assert(row < 3 && col <3);
		return _data[col*3+row];
	}

	double& operator()( size_t row, size_t col ) {
		assert(row < 3 && col < 3);
		return _data[col*3+row];
	}

	matrix3f operator-() const {
		return matrix3f(
			-_data[0], -_data[1], -_data[2],
			-_data[3], -_data[4], -_data[5],
			-_data[6], -_data[7], -_data[8]);
	}

	matrix3f& operator*=( const matrix3f&rhs) {
		return operator=(operator *(rhs));
	}

	matrix3f& operator*=( double rhs) {
		for (int i=0; i<9; i++)
			_data[i] *= rhs;

		return *this;
	}

	matrix3f& operator+=( const matrix3f& rhs ) {
		for (int i=0; i<9; i++)
			_data[i] += rhs._data[i];

		return *this;
	}

	matrix3f& operator-=( const matrix3f& rhs) {
		for (int i=0; i<9; i++)
			_data[i] -= rhs._data[i];

		return *this;
	}

	matrix3f operator*( const matrix3f&rhs) const {
		matrix3f result;
		for ( int r = 0; r < 3; ++r ) {
			for ( int c = 0; c < 3; ++c ) {
				double val = 0;
				for ( int i = 0; i < 3; ++i ) {
					val += operator()( r, i ) * rhs( i, c );
				}
				result( r, c ) = val;
			}
		}
		return result;
	}

	vec3f operator*( const vec3f& rhs) const {
		// _data[ r+c*3 ]
		return vec3f(
			_data[ 0+0*3 ]*rhs.x + _data[ 0+1*3 ]*rhs.y + _data[ 0+2*3 ]*rhs.z,
			_data[ 1+0*3 ]*rhs.x + _data[ 1+1*3 ]*rhs.y + _data[ 1+2*3 ]*rhs.z,
			_data[ 2+0*3 ]*rhs.x + _data[ 2+1*3 ]*rhs.y + _data[ 2+2*3 ]*rhs.z);
	}

	matrix3f operator*( double rhs) const {
		matrix3f tmp(*this);
		tmp *= rhs;
		return tmp;
	}

	matrix3f operator+( const matrix3f& rhs) const {
		matrix3f tmp(*this);
		tmp += rhs;
		return tmp;
	}

	matrix3f operator-( const matrix3f& rhs) const {
		matrix3f tmp(*this);
		tmp -= rhs;
		return tmp;
	}

	bool operator==( const matrix3f& rhs) const {
		bool ret = true;
		for (int i=0; i<9; i++)
			ret = ret && isEqual(_data[i], rhs._data[i]);
		return ret;
	}

	bool operator!=( const matrix3f& rhs) const {
		return !operator==(rhs);
	}

    //! Sum of diagonal elements.
	double getTrace() const {
		return _data[0] + _data[4] + _data[8];
	}

    //! Not the standard definition... max of all elements
	double infinityNorm() const {
		return fmax(
			fmax(
				fmax(
					fmax( _data[ 0 ], _data[ 1 ] ),
					fmax( _data[ 2 ], _data[ 3 ] )
				),
				fmax(
					fmax( _data[ 4 ], _data[ 5 ] ),
					fmax( _data[ 6 ], _data[ 7 ] )
				)
			),
			_data[ 8 ]
		);
	}

    //! Retrieve data as a flat array, column-major storage
	const double* asColMajor() const {
		return _data;
	}

	matrix3f getTranspose() const {
		return matrix3f::rowMajor(_data);
	}

	matrix3f getInverse() const {
		matrix3f result(
			operator()(1,1) * operator()(2,2) - operator()(1,2) * operator()(2,1),
			operator()(0,2) * operator()(2,1) - operator()(0,1) * operator()(2,2),
			operator()(0,1) * operator()(1,2) - operator()(0,2) * operator()(1,1),

			operator()(1,2) * operator()(2,0) - operator()(1,0) * operator()(2,2),
			operator()(0,0) * operator()(2,2) - operator()(0,2) * operator()(2,0),
			operator()(0,2) * operator()(1,0) - operator()(0,0) * operator()(1,2),

			operator()(1,0) * operator()(2,1) - operator()(1,1) * operator()(2,0),
			operator()(0,1) * operator()(2,0) - operator()(0,0) * operator()(2,1),
			operator()(0,0) * operator()(1,1) - operator()(0,1) * operator()(1,0)
		);

		double det =
			operator()(0,0) * result(0,0) +
			operator()(0,1) * result(1,0) +
			operator()(0,2) * result(2,0);

		assert( ! isEqual(det, 0) );

		double invDet = 1.0f / det;
		for( int i = 0; i < 9; ++i )
			result._data[ i ] *= invDet;

		return result;
	}

	double determinant()
	{
		return 
			operator()(0,0)*(operator()(2,2)*operator()(1,1)-operator()(2,1)*operator()(1,2))-
			operator()(1,0)*(operator()(2,2)*operator()(0,1)-operator()(2,1)*operator()(0,2))+
			operator()(2,0)*(operator()(1,2)*operator()(0,1)-operator()(1,1)*operator()(0,2));
	}
};

//! Scalar-matrix multiplication
inline matrix3f operator*( double lhs, const matrix3f& rhs) {
	return rhs * lhs;
}

//! Multiply row vector by matrix, v^T * M
inline vec3f operator*( const vec3f& lhs, const matrix3f& rhs) {
    return vec3f(
        lhs.x * rhs(0,0) + lhs.y * rhs(1,0) + lhs.z * rhs(2,0),
        lhs.x * rhs(0,1) + lhs.y * rhs(1,1) + lhs.z * rhs(2,1),
        lhs.x * rhs(0,2) + lhs.y * rhs(1,2) + lhs.z * rhs(2,2)
    );
}

#include <ostream>

inline std::ostream& operator<<( std::ostream&out, const matrix3f&m )
{
    out << "M3(" << std::endl;
    out << "  " << m(0,0) << " " << m(0,1) << " " << m(0,2) << std::endl;
    out << "  " << m(1,0) << " " << m(1,1) << " " << m(1,2) << std::endl;
    out << "  " << m(2,0) << " " << m(2,1) << " " << m(2,2) << std::endl;
    out << ")";
    return out;
}