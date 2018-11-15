/*
 * @file matrix_2x2.h
 * @brief 2x2 matrix.
 * @author Fei Zhu, Wei Chen
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_

#include <glm/mat2x2.hpp>

// #include "Physika_Core/Utilities/cuda_utilities.h"
// #include "Physika_Core/Utilities/physika_assert.h"
// #include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/square_matrix.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 * SquareMatrix<Scalar,2> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar,2>
{
public:
	typedef Scalar VarType;

    COMM_FUNC SquareMatrix();
    COMM_FUNC explicit SquareMatrix(Scalar);
    COMM_FUNC SquareMatrix(Scalar x00, Scalar x01, Scalar x10, Scalar x11);
    COMM_FUNC SquareMatrix(const Vector<Scalar,2> &row1, const Vector<Scalar,2> &row2);


    COMM_FUNC SquareMatrix(const SquareMatrix<Scalar,2> &) = default;
    COMM_FUNC ~SquareMatrix() = default;

    COMM_FUNC static unsigned int rows() {return 2;}
    COMM_FUNC static unsigned int cols() {return 2;}

    COMM_FUNC Scalar& operator() (unsigned int i, unsigned int j);
    COMM_FUNC const Scalar& operator() (unsigned int i, unsigned int j) const;

    COMM_FUNC const Vector<Scalar,2> rowVector(unsigned int i) const;
    COMM_FUNC const Vector<Scalar,2> colVector(unsigned int i) const;

    COMM_FUNC const SquareMatrix<Scalar,2> operator+ (const SquareMatrix<Scalar,2> &) const;
    COMM_FUNC SquareMatrix<Scalar,2>& operator+= (const SquareMatrix<Scalar,2> &);
    COMM_FUNC const SquareMatrix<Scalar,2> operator- (const SquareMatrix<Scalar,2> &) const;
    COMM_FUNC SquareMatrix<Scalar,2>& operator-= (const SquareMatrix<Scalar,2> &);

    COMM_FUNC SquareMatrix<Scalar,2>& operator= (const SquareMatrix<Scalar,2> &) = default;

    COMM_FUNC bool operator== (const SquareMatrix<Scalar,2> &) const;
    COMM_FUNC bool operator!= (const SquareMatrix<Scalar,2> &) const;

    COMM_FUNC const SquareMatrix<Scalar,2> operator* (Scalar) const;
    COMM_FUNC SquareMatrix<Scalar,2>& operator*= (Scalar);

    COMM_FUNC const Vector<Scalar,2> operator* (const Vector<Scalar,2> &) const;

    COMM_FUNC const SquareMatrix<Scalar,2> operator* (const SquareMatrix<Scalar,2> &) const;
    COMM_FUNC SquareMatrix<Scalar,2>& operator*= (const SquareMatrix<Scalar,2> &);

    COMM_FUNC const SquareMatrix<Scalar,2> operator/ (Scalar) const;
    COMM_FUNC SquareMatrix<Scalar,2>& operator/= (Scalar);

    COMM_FUNC const SquareMatrix<Scalar, 2> operator- (void) const;

    COMM_FUNC const SquareMatrix<Scalar,2> transpose() const;
    COMM_FUNC const SquareMatrix<Scalar,2> inverse() const;

    COMM_FUNC Scalar determinant() const;
    COMM_FUNC Scalar trace() const;
    COMM_FUNC Scalar doubleContraction(const SquareMatrix<Scalar,2> &) const;//double contraction
    COMM_FUNC Scalar frobeniusNorm() const;

    void singularValueDecomposition(SquareMatrix<Scalar,2> &left_singular_vectors,
                                    Vector<Scalar,2> &singular_values,   //singular values are in descending order
                                    SquareMatrix<Scalar,2> &right_singular_vectors) const;

    void singularValueDecomposition(SquareMatrix<Scalar,2> &left_singular_vectors,
                                    SquareMatrix<Scalar,2> &singular_values_diagonal,   //singular values in descending order as a diagonal matrix
                                    SquareMatrix<Scalar,2> &right_singular_vectors) const;

    void eigenDecomposition(Vector<Scalar,2> &eigen_values_real,
                            Vector<Scalar,2> &eigen_values_imag,
                            SquareMatrix<Scalar,2> &eigen_vectors_real,
                            SquareMatrix<Scalar,2> &eigen_vectors_imag);

    COMM_FUNC static const SquareMatrix<Scalar,2> identityMatrix();

	COMM_FUNC Scalar* getDataPtr() { return &data_[0].x; }

protected:
    glm::tmat2x2<Scalar> data_; //default: zero matrix

private:
    void compileTimeCheck()
    {
        //SquareMatrix<Scalar,Dim> is only defined for element type of integer and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value || is_floating_point<Scalar>::value),
                              "SquareMatrix<Scalar,2> are only defined for integer types and floating-point types.");
    }

};

//overriding << for SquareMatrix<Scalar,2>
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const SquareMatrix<Scalar,2> &mat)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
    {
        s<<"["<<static_cast<int>(mat(0,0))<<", "<<static_cast<int>(mat(0,1))<<"; ";
        s<<static_cast<int>(mat(1,0))<<", "<<static_cast<int>(mat(1,1))<<"]";
    }
    else
    {
        s<<"["<<mat(0,0)<<", "<<mat(0,1)<<"; ";
        s<<mat(1,0)<<", "<<mat(1,1)<<"]";
    }
    return s;
}

//make * operator commutative
template <typename S, typename T>
COMM_FUNC const SquareMatrix<T,2> operator* (S scale, const SquareMatrix<T,2> &mat)
{
    return mat*scale;
}

template class SquareMatrix<float, 2>;
template class SquareMatrix<double, 2>;
//convenient typedefs
typedef SquareMatrix<float,2> Matrix2f;
typedef SquareMatrix<double,2> Matrix2d;
//typedef SquareMatrix<int, 2> Matrix2i;

}  //end of namespace Physika

#include "matrix_2x2.inl"
#endif //PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
