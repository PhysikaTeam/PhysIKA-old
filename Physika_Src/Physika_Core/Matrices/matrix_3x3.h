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
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

template <typename Scalar>
class Matrix3x3: public MatrixBase
{
public:
    Matrix3x3();
    Matrix3x3(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21, Scalar x22);
    Matrix3x3(const Matrix3x3<Scalar>&);
    ~Matrix3x3();
    inline int rows() const{return 3;}
    inline int cols() const{return 3;}
    Scalar& operator() (int i, int j );
    const Scalar& operator() (int i, int j) const;
    Matrix3x3<Scalar> operator+ (const Matrix3x3<Scalar> &) const;
    Matrix3x3<Scalar>& operator+= (const Matrix3x3<Scalar> &);
    Matrix3x3<Scalar> operator- (const Matrix3x3<Scalar> &) const;
    Matrix3x3<Scalar>& operator-= (const Matrix3x3<Scalar> &);
    Matrix3x3<Scalar>& operator= (const Matrix3x3<Scalar> &);
    bool operator== (const Matrix3x3<Scalar> &) const;
    Matrix3x3<Scalar> operator* (Scalar) const;
    Matrix3x3<Scalar>& operator*= (Scalar);
    Matrix3x3<Scalar> operator/ (Scalar) const;
    Matrix3x3<Scalar>& operator/= (Scalar);
    Matrix3x3<Scalar> transpose() const;
    Matrix3x3<Scalar> inverse() const;
    Scalar determinant() const;
 
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,3,3> eigen_matrix_3x3_;
#endif

};

//overriding << for Matrix3x3
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Matrix3x3<Scalar> &mat)
{
    s<<mat(0,0)<<", "<<mat(0,1)<<", "<<mat(0,2)<<std::endl;
    s<<mat(1,0)<<", "<<mat(1,1)<<", "<<mat(1,2)<<std::endl;
    s<<mat(2,0)<<", "<<mat(2,1)<<", "<<mat(2,2)<<std::endl;
    return s;
}
 
//make * operator commutative
template <typename S, typename T>
Matrix3x3<T> operator* (S scale, const Matrix3x3<T> &mat)
{
    return mat*scale;
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_3X3_H_
