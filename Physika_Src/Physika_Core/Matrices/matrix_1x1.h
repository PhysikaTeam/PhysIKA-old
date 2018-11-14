/*
 * @file matrix_1x1.h
 * @brief 1x1 matrix.
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

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_1X1_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_1X1_H_
// #include "Physika_Core/Utilities/physika_assert.h"
// #include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/square_matrix.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 * SquareMatrix<Scalar,1> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar,1>
{
public:
	typedef Scalar VarType;

    COMM_FUNC SquareMatrix();
    COMM_FUNC explicit SquareMatrix(Scalar);
    COMM_FUNC SquareMatrix(const SquareMatrix<Scalar,1> &) = default;
    COMM_FUNC ~SquareMatrix() = default;

    COMM_FUNC static unsigned int rows() {return 1;}
    COMM_FUNC static unsigned int cols() {return 1;}

    COMM_FUNC Scalar& operator() (unsigned int i, unsigned int j);
    COMM_FUNC const Scalar& operator() (unsigned int i, unsigned int j) const;

    COMM_FUNC const Vector<Scalar,1> rowVector(unsigned int i) const;
    COMM_FUNC const Vector<Scalar,1> colVector(unsigned int i) const;

    COMM_FUNC const SquareMatrix<Scalar,1> operator+ (const SquareMatrix<Scalar,1> &) const;
    COMM_FUNC SquareMatrix<Scalar,1>& operator+= (const SquareMatrix<Scalar,1> &);
    COMM_FUNC const SquareMatrix<Scalar,1> operator- (const SquareMatrix<Scalar,1> &) const;
    COMM_FUNC SquareMatrix<Scalar,1>& operator-= (const SquareMatrix<Scalar,1> &);

    COMM_FUNC SquareMatrix<Scalar,1>& operator= (const SquareMatrix<Scalar,1> &) = default;

    COMM_FUNC bool operator== (const SquareMatrix<Scalar,1> &) const;
    COMM_FUNC bool operator!= (const SquareMatrix<Scalar,1> &) const;

    COMM_FUNC const SquareMatrix<Scalar,1> operator* (Scalar) const;
    COMM_FUNC SquareMatrix<Scalar,1>& operator*= (Scalar);
    
    COMM_FUNC const Vector<Scalar,1> operator* (const Vector<Scalar,1> &) const;
    COMM_FUNC const SquareMatrix<Scalar,1> operator* (const SquareMatrix<Scalar,1> &) const;
    COMM_FUNC SquareMatrix<Scalar, 1> & operator *= (const SquareMatrix<Scalar, 1> &);

    COMM_FUNC const SquareMatrix<Scalar,1> operator/ (Scalar) const;
    COMM_FUNC SquareMatrix<Scalar,1>& operator/= (Scalar);

    COMM_FUNC const SquareMatrix<Scalar, 1> operator- (void) const;

    COMM_FUNC const SquareMatrix<Scalar,1> transpose() const;
    COMM_FUNC const SquareMatrix<Scalar,1> inverse() const;

    COMM_FUNC Scalar determinant() const;
    COMM_FUNC Scalar trace() const;
    COMM_FUNC Scalar doubleContraction(const SquareMatrix<Scalar,1> &) const;//double contraction
    COMM_FUNC Scalar frobeniusNorm() const;

    COMM_FUNC static const SquareMatrix<Scalar,1> identityMatrix();

	COMM_FUNC Scalar* getDataPtr() { return &data_; }

protected:
    Scalar data_; //default:zero matrix

private:
    void compileTimeCheck()
    {
        //SquareMatrix<Scalar,Dim> is only defined for element type of integers and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "SquareMatrix<Scalar,1> are only defined for integer types and floating-point types.");
    }

};

//overriding << for SquareMatrix<Scalar,1>
// template <typename Scalar>
// inline std::ostream& operator<< (std::ostream &s, const SquareMatrix<Scalar,1> &mat)
// {
//     if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
//     {
//         s<<"["<<static_cast<int>(mat(0,0))<<"]";
//     }
//     else
//     {
//         s<<"["<<mat(0,0)<<"]";
//     }
//     return s;
// }

//make * operator commutative
// template <typename S, typename T>
// COMM_FUNC  const SquareMatrix<T,1> operator* (S scale, const SquareMatrix<T,1> &mat)
// {
//     return mat*scale;
// }

template class SquareMatrix<float, 1>;
template class SquareMatrix<double, 1>;
//convenient typedefs
typedef SquareMatrix<float,1> Matrix1f;
typedef SquareMatrix<double,1> Matrix1d;
//typedef SquareMatrix<int,1> Matrix1i;

}  //end of namespace Physika

#include "matrix_1x1.inl"
#endif //PHYSIKA_CORE_MATRICES_MATRIX_1X1_H_
