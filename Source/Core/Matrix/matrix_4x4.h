/*
 * @file matrix_4x4.h
 * @brief 4x4 matrix.
 * @author Sheng Yang, Fei Zhu, Liyou Xu
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_4X4_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_4X4_H_

#include <glm/mat4x4.hpp>

// #include "Core/Utilities/physika_assert.h"
// #include "Core/Utilities/cuda_utilities.h"
// #include "Core/Utilities/type_utilities.h"
#include "square_matrix.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class Vector;

/*
 * SquareMatrix<Scalar,4> are defined for C++ fundamental integer and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar, 4>
{
public:
    typedef Scalar VarType;

    COMM_FUNC SquareMatrix();
    COMM_FUNC explicit SquareMatrix(Scalar);
    COMM_FUNC SquareMatrix(Scalar x00, Scalar x01, Scalar x02, Scalar x03, Scalar x10, Scalar x11, Scalar x12, Scalar x13, Scalar x20, Scalar x21, Scalar x22, Scalar x23, Scalar x30, Scalar x31, Scalar x32, Scalar x33);
    COMM_FUNC SquareMatrix(const Vector<Scalar, 4>& row1, const Vector<Scalar, 4>& row2, const Vector<Scalar, 4>& row3, const Vector<Scalar, 4>& row4);

    COMM_FUNC SquareMatrix(const SquareMatrix<Scalar, 4>&);
    COMM_FUNC ~SquareMatrix();

    COMM_FUNC static unsigned int rows()
    {
        return 4;
    }
    COMM_FUNC static unsigned int cols()
    {
        return 4;
    }

    COMM_FUNC Scalar& operator()(unsigned int i, unsigned int j);
    COMM_FUNC const Scalar& operator()(unsigned int i, unsigned int j) const;

    COMM_FUNC const Vector<Scalar, 4> row(unsigned int i) const;
    COMM_FUNC const Vector<Scalar, 4> col(unsigned int i) const;

    COMM_FUNC void setRow(unsigned int i, const Vector<Scalar, 4>& vec);
    COMM_FUNC void setCol(unsigned int j, const Vector<Scalar, 4>& vec);

    COMM_FUNC const SquareMatrix<Scalar, 4> operator+(const SquareMatrix<Scalar, 4>&) const;
    COMM_FUNC SquareMatrix<Scalar, 4>& operator+=(const SquareMatrix<Scalar, 4>&);
    COMM_FUNC const SquareMatrix<Scalar, 4> operator-(const SquareMatrix<Scalar, 4>&) const;
    COMM_FUNC SquareMatrix<Scalar, 4>& operator-=(const SquareMatrix<Scalar, 4>&);
    COMM_FUNC const SquareMatrix<Scalar, 4> operator*(const SquareMatrix<Scalar, 4>&) const;
    COMM_FUNC SquareMatrix<Scalar, 4>& operator*=(const SquareMatrix<Scalar, 4>&);
    COMM_FUNC const SquareMatrix<Scalar, 4> operator/(const SquareMatrix<Scalar, 4>&) const;
    COMM_FUNC SquareMatrix<Scalar, 4>& operator/=(const SquareMatrix<Scalar, 4>&);

    COMM_FUNC SquareMatrix<Scalar, 4>& operator=(const SquareMatrix<Scalar, 4>&);

    COMM_FUNC bool operator==(const SquareMatrix<Scalar, 4>&) const;
    COMM_FUNC bool operator!=(const SquareMatrix<Scalar, 4>&) const;

    COMM_FUNC const SquareMatrix<Scalar, 4> operator*(const Scalar&) const;
    COMM_FUNC SquareMatrix<Scalar, 4>& operator*=(const Scalar&);
    COMM_FUNC const SquareMatrix<Scalar, 4> operator/(const Scalar&) const;
    COMM_FUNC SquareMatrix<Scalar, 4>& operator/=(const Scalar&);

    COMM_FUNC const Vector<Scalar, 4> operator*(const Vector<Scalar, 4>&) const;

    COMM_FUNC const SquareMatrix<Scalar, 4> operator-(void) const;

    COMM_FUNC const SquareMatrix<Scalar, 4> transpose() const;
    COMM_FUNC const SquareMatrix<Scalar, 4> inverse() const;

    COMM_FUNC Scalar determinant() const;
    COMM_FUNC Scalar trace() const;
    COMM_FUNC Scalar doubleContraction(const SquareMatrix<Scalar, 4>&) const;  //double contraction
    COMM_FUNC Scalar frobeniusNorm() const;
    COMM_FUNC Scalar oneNorm() const;
    COMM_FUNC Scalar infNorm() const;

    COMM_FUNC static const SquareMatrix<Scalar, 4> identityMatrix();

    COMM_FUNC Scalar* getDataPtr()
    {
        return &data_[0].x;
    }

protected:
    glm::tmat4x4<Scalar> data_;  //default: zero matrix
};

template class SquareMatrix<float, 4>;
template class SquareMatrix<double, 4>;
//convenient typedefs
typedef SquareMatrix<float, 4>  Matrix4f;
typedef SquareMatrix<double, 4> Matrix4d;
//typedef SquareMatrix<int,4> Matrix4i;

}  //end of namespace PhysIKA

#include "matrix_4x4.inl"
#endif  //PHYSIKA_CORE_MATRICES_MATRIX_4X4_H_
