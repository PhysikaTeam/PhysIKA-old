/*
 * @file matrix_2x2.h
 * @brief 2x2 matrix.
 * @author Fei Zhu, Wei Chen
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_

#include <glm/mat2x2.hpp>

// #include "Core/Utilities/cuda_utilities.h"
// #include "Core/Utilities/physika_assert.h"
// #include "Core/Utilities/type_utilities.h"
#include "square_matrix.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class Vector;

/*
 * SquareMatrix<Scalar,2> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar, 2>
{
public:
    typedef Scalar VarType;

    COMM_FUNC SquareMatrix();
    COMM_FUNC explicit SquareMatrix(Scalar);
    COMM_FUNC SquareMatrix(Scalar x00, Scalar x01, Scalar x10, Scalar x11);
    COMM_FUNC SquareMatrix(const Vector<Scalar, 2>& row1, const Vector<Scalar, 2>& row2);

    COMM_FUNC SquareMatrix(const SquareMatrix<Scalar, 2>&);
    COMM_FUNC ~SquareMatrix();

    COMM_FUNC static unsigned int rows()
    {
        return 2;
    }
    COMM_FUNC static unsigned int cols()
    {
        return 2;
    }

    COMM_FUNC Scalar& operator()(unsigned int i, unsigned int j);
    COMM_FUNC const Scalar& operator()(unsigned int i, unsigned int j) const;

    COMM_FUNC const Vector<Scalar, 2> row(unsigned int i) const;
    COMM_FUNC const Vector<Scalar, 2> col(unsigned int i) const;

    COMM_FUNC void setRow(unsigned int i, const Vector<Scalar, 2>& vec);
    COMM_FUNC void setCol(unsigned int j, const Vector<Scalar, 2>& vec);

    COMM_FUNC const SquareMatrix<Scalar, 2> operator+(const SquareMatrix<Scalar, 2>&) const;
    COMM_FUNC SquareMatrix<Scalar, 2>& operator+=(const SquareMatrix<Scalar, 2>&);
    COMM_FUNC const SquareMatrix<Scalar, 2> operator-(const SquareMatrix<Scalar, 2>&) const;
    COMM_FUNC SquareMatrix<Scalar, 2>& operator-=(const SquareMatrix<Scalar, 2>&);
    COMM_FUNC const SquareMatrix<Scalar, 2> operator*(const SquareMatrix<Scalar, 2>&) const;
    COMM_FUNC SquareMatrix<Scalar, 2>& operator*=(const SquareMatrix<Scalar, 2>&);
    COMM_FUNC const SquareMatrix<Scalar, 2> operator/(const SquareMatrix<Scalar, 2>&) const;
    COMM_FUNC SquareMatrix<Scalar, 2>& operator/=(const SquareMatrix<Scalar, 2>&);

    COMM_FUNC SquareMatrix<Scalar, 2>& operator=(const SquareMatrix<Scalar, 2>&);

    COMM_FUNC bool operator==(const SquareMatrix<Scalar, 2>&) const;
    COMM_FUNC bool operator!=(const SquareMatrix<Scalar, 2>&) const;

    COMM_FUNC const SquareMatrix<Scalar, 2> operator*(const Scalar&) const;
    COMM_FUNC SquareMatrix<Scalar, 2>& operator*=(const Scalar&);
    COMM_FUNC const SquareMatrix<Scalar, 2> operator/(const Scalar&) const;
    COMM_FUNC SquareMatrix<Scalar, 2>& operator/=(const Scalar&);

    COMM_FUNC const Vector<Scalar, 2> operator*(const Vector<Scalar, 2>&) const;

    COMM_FUNC const SquareMatrix<Scalar, 2> operator-(void) const;

    COMM_FUNC const SquareMatrix<Scalar, 2> transpose() const;
    COMM_FUNC const SquareMatrix<Scalar, 2> inverse() const;

    COMM_FUNC Scalar determinant() const;
    COMM_FUNC Scalar trace() const;
    COMM_FUNC Scalar doubleContraction(const SquareMatrix<Scalar, 2>&) const;  //double contraction
    COMM_FUNC Scalar frobeniusNorm() const;
    COMM_FUNC Scalar oneNorm() const;
    COMM_FUNC Scalar infNorm() const;

    COMM_FUNC static const SquareMatrix<Scalar, 2> identityMatrix();

    COMM_FUNC Scalar* getDataPtr()
    {
        return &data_[0].x;
    }

protected:
    glm::tmat2x2<Scalar> data_;  //default: zero matrix
};

//make * operator commutative
template <typename S, typename T>
COMM_FUNC const SquareMatrix<T, 2> operator*(S scale, const SquareMatrix<T, 2>& mat)
{
    return mat * scale;
}

template class SquareMatrix<float, 2>;
template class SquareMatrix<double, 2>;
//convenient typedefs
typedef SquareMatrix<float, 2>  Matrix2f;
typedef SquareMatrix<double, 2> Matrix2d;
//typedef SquareMatrix<int, 2> Matrix2i;

}  //end of namespace PhysIKA

#include "matrix_2x2.inl"
#endif  //PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
