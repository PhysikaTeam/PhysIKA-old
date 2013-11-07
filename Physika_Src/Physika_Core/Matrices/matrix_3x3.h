/*
 * @file matrix_3x3.h 
 * @brief 3x3 matrix.
 * @author Sheng Yang, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_3X3_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_3X3_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/square_matrix.h"

namespace Physika{

template <typename Scalar>
class SquareMatrix<Scalar,3>: public MatrixBase
{
public:
    SquareMatrix();
    SquareMatrix(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21, Scalar x22);
    SquareMatrix(const Vector<Scalar,3> &row1, const Vector<Scalar,3> &row2, const Vector<Scalar,3> &row3);
    SquareMatrix(const SquareMatrix<Scalar,3>&);
    ~SquareMatrix();
    inline int rows() const{return 3;}
    inline int cols() const{return 3;}
    Scalar& operator() (int i, int j );
    const Scalar& operator() (int i, int j) const;
    SquareMatrix<Scalar,3> operator+ (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3>& operator+= (const SquareMatrix<Scalar,3> &);
    SquareMatrix<Scalar,3> operator- (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3>& operator-= (const SquareMatrix<Scalar,3> &);
    SquareMatrix<Scalar,3>& operator= (const SquareMatrix<Scalar,3> &);
    bool operator== (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3> operator* (Scalar) const;
    SquareMatrix<Scalar,3>& operator*= (Scalar);
    Vector<Scalar,3> operator* (const Vector<Scalar,3> &) const;
    SquareMatrix<Scalar,3> operator* (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3> operator/ (Scalar) const;
    SquareMatrix<Scalar,3>& operator/= (Scalar);
    SquareMatrix<Scalar,3> transpose() const;
    SquareMatrix<Scalar,3> inverse() const;
    Scalar determinant() const;
    Scalar trace() const;
    Scalar doubleContraction(const SquareMatrix<Scalar,3> &) const;//double contraction
    static SquareMatrix<Scalar,3> identityMatrix();
 
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,3,3> eigen_matrix_3x3_;
#endif

};

//overriding << for SquareMatrix<Scalar,3>
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const SquareMatrix<Scalar,3> &mat)
{
    s<<mat(0,0)<<", "<<mat(0,1)<<", "<<mat(0,2)<<std::endl;
    s<<mat(1,0)<<", "<<mat(1,1)<<", "<<mat(1,2)<<std::endl;
    s<<mat(2,0)<<", "<<mat(2,1)<<", "<<mat(2,2)<<std::endl;
    return s;
}
 
//make * operator commutative
template <typename S, typename T>
SquareMatrix<T,3> operator* (S scale, const SquareMatrix<T,3> &mat)
{
    return mat*scale;
}

//convenient typedefs
#define Matrix3x3(Scalar) SquareMatrix<Scalar,3>
typedef SquareMatrix<float,3> Matrix3f;
typedef SquareMatrix<double,3> Matrix3d;

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_3X3_H_
