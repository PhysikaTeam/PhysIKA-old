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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
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
    CPU_GPU_FUNC_DECL SquareMatrix();
    CPU_GPU_FUNC_DECL explicit SquareMatrix(Scalar);
    CPU_GPU_FUNC_DECL SquareMatrix(const SquareMatrix<Scalar,1> &) = default;
    CPU_GPU_FUNC_DECL ~SquareMatrix() = default;

    CPU_GPU_FUNC_DECL static unsigned int rows() {return 1;}
    CPU_GPU_FUNC_DECL static unsigned int cols() {return 1;}

    CPU_GPU_FUNC_DECL Scalar& operator() (unsigned int i, unsigned int j);
    CPU_GPU_FUNC_DECL const Scalar& operator() (unsigned int i, unsigned int j) const;

    CPU_GPU_FUNC_DECL const Vector<Scalar,1> rowVector(unsigned int i) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar,1> colVector(unsigned int i) const;

    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> operator+ (const SquareMatrix<Scalar,1> &) const;
    CPU_GPU_FUNC_DECL SquareMatrix<Scalar,1>& operator+= (const SquareMatrix<Scalar,1> &);
    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> operator- (const SquareMatrix<Scalar,1> &) const;
    CPU_GPU_FUNC_DECL SquareMatrix<Scalar,1>& operator-= (const SquareMatrix<Scalar,1> &);

    CPU_GPU_FUNC_DECL SquareMatrix<Scalar,1>& operator= (const SquareMatrix<Scalar,1> &) = default;

    CPU_GPU_FUNC_DECL bool operator== (const SquareMatrix<Scalar,1> &) const;
    CPU_GPU_FUNC_DECL bool operator!= (const SquareMatrix<Scalar,1> &) const;

    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> operator* (Scalar) const;
    CPU_GPU_FUNC_DECL SquareMatrix<Scalar,1>& operator*= (Scalar);
    
    CPU_GPU_FUNC_DECL const Vector<Scalar,1> operator* (const Vector<Scalar,1> &) const;
    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> operator* (const SquareMatrix<Scalar,1> &) const;
    CPU_GPU_FUNC_DECL SquareMatrix<Scalar, 1> & operator *= (const SquareMatrix<Scalar, 1> &);

    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> operator/ (Scalar) const;
    CPU_GPU_FUNC_DECL SquareMatrix<Scalar,1>& operator/= (Scalar);

    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar, 1> operator- (void) const;

    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> transpose() const;
    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> inverse() const;

    CPU_GPU_FUNC_DECL Scalar determinant() const;
    CPU_GPU_FUNC_DECL Scalar trace() const;
    CPU_GPU_FUNC_DECL Scalar doubleContraction(const SquareMatrix<Scalar,1> &) const;//double contraction
    CPU_GPU_FUNC_DECL Scalar frobeniusNorm() const;

    CPU_GPU_FUNC_DECL static const SquareMatrix<Scalar,1> identityMatrix();

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
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const SquareMatrix<Scalar,1> &mat)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
    {
        s<<"["<<static_cast<int>(mat(0,0))<<"]";
    }
    else
    {
        s<<"["<<mat(0,0)<<"]";
    }
    return s;
}

//make * operator commutative
template <typename S, typename T>
CPU_GPU_FUNC_DECL  const SquareMatrix<T,1> operator* (S scale, const SquareMatrix<T,1> &mat)
{
    return mat*scale;
}

//convenient typedefs
typedef SquareMatrix<float,1> Matrix1f;
typedef SquareMatrix<double,1> Matrix1d;
typedef SquareMatrix<int,1> Matrix1i;

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_1X1_H_
