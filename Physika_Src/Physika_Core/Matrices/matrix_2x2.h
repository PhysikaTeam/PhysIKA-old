/*
 * @file matrix_2x2.h 
 * @brief 2x2 matrix.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

template <typename Scalar>
class Matrix2x2: public MatrixBase<Scalar,2,2>
{
public:
  Matrix2x2();
  Matrix2x2(Scalar x00, Scalar x01, Scalar x10, Scalar x11);
  ~Matrix2x2();
  inline int rows() const{return 2;}
  inline int cols() const{return 2;}
  Scalar& operator() (int i, int j);
  const Scalar& operator() (int i, int j) const;
  Matrix2x2<Scalar> operator+ (const Matrix2x2<Scalar> &) const;
  Matrix2x2<Scalar>& operator+= (const Matrix2x2<Scalar> &);
  Matrix2x2<Scalar> operator- (const Matrix2x2<Scalar> &) const;
  Matrix2x2<Scalar>& operator-= (const Matrix2x2<Scalar> &);
  Matrix2x2<Scalar>& operator= (const Matrix2x2<Scalar> &);
  bool operator== (const Matrix2x2<Scalar> &) const;
  Matrix2x2<Scalar> operator* (Scalar) const;
  Matrix2x2<Scalar>& operator*= (Scalar);
  Matrix2x2<Scalar> operator/ (Scalar) const;
  Matrix2x2<Scalar>& operator/= (Scalar);
  Matrix2x2<Scalar> transpose() const;
  Matrix2x2<Scalar> inverse() const;
  Scalar determinant() const;
 
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  Eigen::Matrix<Scalar,2,2> eigen_matrix_2x2_;
#endif

};

//overriding << for Matrix2x2
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Matrix2x2<Scalar> &mat)
{
  s<<mat(0,0)<<", "<<mat(0,1)<<std::endl;
  s<<mat(1,0)<<", "<<mat(1,1)<<std::endl;
  return s;
}
 
}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
