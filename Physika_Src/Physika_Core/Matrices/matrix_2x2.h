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

#ifndef PHYSIKA_MATRICES_MATRIX_2X2_H_
#define PHYSIKA_MATRICES_MATRIX_2X2_H_

#include <iostream>
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

template <typename Scalar>
class Matrix2x2: public MatrixBase<Scalar,2,2>
{
public:
  Matrix2x2();
  Matrix2x2(const Scalar x00, const Scalar x01, const Scalar x10, const Scalar x11);
  ~Matrix2x2();
  inline int rows const(){return 2};
  inline int cols const(){return 2};
  Scalar& operator() (const int i, const int j);
  Matrix2x2<Scalar> operator+ (const Matrix2x2<Scalar> &);
  Matrix2x2<Scalar> operator- (const Matrix2x2<Scalar> &);
  Matrix2x2<Scalar>& operator= (const Matrix2x2<Scalar> &);
  Matrix2x2<Scalar> operator* (const Scalar);
  Matrix2x2<Scalar> operator/ (const Scalar);
  Matrix2x2<Scalar> transpose();
  Matrix2x2<Scalar> inverse();

  friend std::ostream &operator<< (std::ostream &, const Matrix2x2<Scalar> &);
 
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
  Eigen::Matrix<Scalar,2,2> eigen_matrix_2x2_;
#endif

};
 
}  //end of namespace Physika

#endif //PHYSIKA_MATRICES_MATRIX_2X2_H_
