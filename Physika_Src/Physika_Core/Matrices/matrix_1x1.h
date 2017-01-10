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

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/square_matrix.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 * SquareMatrix<Scalar,1> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar,1>: public MatrixBase
{
public:
    SquareMatrix();
    explicit SquareMatrix(Scalar);
    SquareMatrix(const SquareMatrix<Scalar,1> &) = default;
    ~SquareMatrix() = default;

    inline unsigned int rows() const{return 1;}
    inline unsigned int cols() const{return 1;}

    Scalar& operator() (unsigned int i, unsigned int j);
    const Scalar& operator() (unsigned int i, unsigned int j) const;

    const Vector<Scalar,1> rowVector(unsigned int i) const;
    const Vector<Scalar,1> colVector(unsigned int i) const;

    const SquareMatrix<Scalar,1> operator+ (const SquareMatrix<Scalar,1> &) const;
    SquareMatrix<Scalar,1>& operator+= (const SquareMatrix<Scalar,1> &);
    const SquareMatrix<Scalar,1> operator- (const SquareMatrix<Scalar,1> &) const;
    SquareMatrix<Scalar,1>& operator-= (const SquareMatrix<Scalar,1> &);

    SquareMatrix<Scalar,1>& operator= (const SquareMatrix<Scalar,1> &) = default;

    bool operator== (const SquareMatrix<Scalar,1> &) const;
    bool operator!= (const SquareMatrix<Scalar,1> &) const;

    const SquareMatrix<Scalar,1> operator* (Scalar) const;
    SquareMatrix<Scalar,1>& operator*= (Scalar);
    
    const Vector<Scalar,1> operator* (const Vector<Scalar,1> &) const;
    const SquareMatrix<Scalar,1> operator* (const SquareMatrix<Scalar,1> &) const;
    SquareMatrix<Scalar, 1> & operator *= (const SquareMatrix<Scalar, 1> &);

    const SquareMatrix<Scalar,1> operator/ (Scalar) const;
    SquareMatrix<Scalar,1>& operator/= (Scalar);

    const SquareMatrix<Scalar,1> transpose() const;
    const SquareMatrix<Scalar,1> inverse() const;

    Scalar determinant() const;
    Scalar trace() const;
    Scalar doubleContraction(const SquareMatrix<Scalar,1> &) const;//double contraction
    Scalar frobeniusNorm() const;

    static const SquareMatrix<Scalar,1> identityMatrix();

protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,1,1> eigen_matrix_1x1_; //default:zero matrix
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar data_; //default:zero matrix
#endif
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
inline const SquareMatrix<T,1> operator* (S scale, const SquareMatrix<T,1> &mat)
{
    return mat*scale;
}

//convenient typedefs
typedef SquareMatrix<float,1> Matrix1f;
typedef SquareMatrix<double,1> Matrix1d;
typedef SquareMatrix<int,1> Matrix1i;

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_1X1_H_
