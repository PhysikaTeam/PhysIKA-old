/*
 * @file matrix_4x4.h 
 * @brief 4x4 matrix.
 * @author Sheng Yang, Fei Zhu, Liyou Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_4X4_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_4X4_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Matrices/square_matrix.h"

namespace Physika{

/*
 * SquareMatrix<Scalar,4> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar,4>: public MatrixBase
{
public:
    SquareMatrix();
    SquareMatrix(Scalar x00, Scalar x01, Scalar x02, Scalar x03, Scalar x10, Scalar x11, Scalar x12, Scalar x13, Scalar x20, Scalar x21, Scalar x22, Scalar x23, Scalar x30, Scalar x31, Scalar x32, Scalar x33);
    SquareMatrix(const Vector<Scalar,4> &row1, const Vector<Scalar,4> &row2, const Vector<Scalar,4> &row3, const Vector<Scalar,4> &row4);
    SquareMatrix(const SquareMatrix<Scalar,4>&);
    ~SquareMatrix();
    inline int rows() const{return 4;}
    inline int cols() const{return 4;}
    Scalar& operator() (int i, int j );
    const Scalar& operator() (int i, int j) const;
    SquareMatrix<Scalar,4> operator+ (const SquareMatrix<Scalar,4> &) const;
    SquareMatrix<Scalar,4>& operator+= (const SquareMatrix<Scalar,4> &);
    SquareMatrix<Scalar,4> operator- (const SquareMatrix<Scalar,4> &) const;
    SquareMatrix<Scalar,4>& operator-= (const SquareMatrix<Scalar,4> &);
    SquareMatrix<Scalar,4>& operator= (const SquareMatrix<Scalar,4> &);
    bool operator== (const SquareMatrix<Scalar,4> &) const;
    bool operator!= (const SquareMatrix<Scalar,4> &) const;
    SquareMatrix<Scalar,4> operator* (Scalar) const;
    SquareMatrix<Scalar,4>& operator*= (Scalar);
    Vector<Scalar,4> operator* (const Vector<Scalar,4> &) const;
    SquareMatrix<Scalar,4> operator* (const SquareMatrix<Scalar,4> &) const;
    SquareMatrix<Scalar,4> operator/ (Scalar) const;
    SquareMatrix<Scalar,4>& operator/= (Scalar);
    SquareMatrix<Scalar,4> transpose() const;
    SquareMatrix<Scalar,4> inverse() const;
    Scalar determinant() const;
    Scalar trace() const;
    Scalar doubleContraction(const SquareMatrix<Scalar,4> &) const;//double contraction
    static SquareMatrix<Scalar,4> identityMatrix();
 
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,4,4> eigen_matrix_4x4_;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar data_[4][4];
#endif

};

//overriding << for SquareMatrix<Scalar,4>
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const SquareMatrix<Scalar,4> &mat)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
    {
        s<<"[";
        s<<static_cast<int>(mat(0,0))<<", "<<static_cast<int>(mat(0,1))<<", "<<static_cast<int>(mat(0,2))<<", "<<static_cast<int>(mat(0,3))<<"; ";
        s<<static_cast<int>(mat(1,0))<<", "<<static_cast<int>(mat(1,1))<<", "<<static_cast<int>(mat(1,2))<<", "<<static_cast<int>(mat(1,3))<<"; ";
        s<<static_cast<int>(mat(2,0))<<", "<<static_cast<int>(mat(2,1))<<", "<<static_cast<int>(mat(2,2))<<", "<<static_cast<int>(mat(2,3))<<"; ";
        s<<static_cast<int>(mat(3,0))<<", "<<static_cast<int>(mat(3,1))<<", "<<static_cast<int>(mat(3,2))<<", "<<static_cast<int>(mat(3,3))<<"]";
    }
    else
    {
        s<<"[";
        s<<mat(0,0)<<", "<<mat(0,1)<<", "<<mat(0,2)<<", "<<mat(0,3)<<"; ";
        s<<mat(1,0)<<", "<<mat(1,1)<<", "<<mat(1,2)<<", "<<mat(1,3)<<"; ";
        s<<mat(2,0)<<", "<<mat(2,1)<<", "<<mat(2,2)<<", "<<mat(2,3)<<"; ";
        s<<mat(3,0)<<", "<<mat(3,1)<<", "<<mat(3,2)<<", "<<mat(3,3)<<"]";
    }
    return s;
}
 
//make * operator commutative
template <typename S, typename T>
SquareMatrix<T,4> operator* (S scale, const SquareMatrix<T,4> &mat)
{
    return mat*scale;
}

//convenient typedefs
typedef SquareMatrix<float,4> Matrix4f;
typedef SquareMatrix<double,4> Matrix4d;
typedef SquareMatrix<int,4> Matrix4i;

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_4X4_H_
