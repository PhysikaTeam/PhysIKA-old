/*
 * @file vector_2d.h 
 * @brief 2d vector.
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

#ifndef PHSYIKA_CORE_VECTORS_VECTOR_2D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_2D_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Vectors/vector_base.h"

namespace Physika{

template <typename Scalar>
class Vector2D: public VectorBase
{
 public:
  Vector2D();
  Vector2D(Scalar x, Scalar y);
  Vector2D(Scalar);
  ~Vector2D();
  inline int dims() const{return 2;}
  Scalar& operator[] (int);
  const Scalar& operator[] (int) const;
  Vector2D<Scalar> operator+ (const Vector2D<Scalar> &) const;
  Vector2D<Scalar>& operator+= (const Vector2D<Scalar> &);
  Vector2D<Scalar> operator- (const Vector2D<Scalar> &) const;
  Vector2D<Scalar>& operator-= (const Vector2D<Scalar> &);
  Vector2D<Scalar>& operator= (const Vector2D<Scalar> &);
  bool operator== (const Vector2D<Scalar> &) const;
  Vector2D<Scalar> operator* (Scalar) const;
  Vector2D<Scalar>& operator*= (Scalar);
  Vector2D<Scalar> operator/ (Scalar) const;
  Vector2D<Scalar>& operator/= (Scalar);
  Scalar norm() const;
  Vector2D<Scalar>& normalize();

 protected:
#ifdef PHYSIKA_USE_EIGEN_VECTOR
  Eigen::Matrix<Scalar,2,1> eigen_vector_2x_;
#endif

};

//overriding << for vector2D
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Vector2D<Scalar> &vec)
{
  s<<vec[0]<<", "<<vec[1]<<std::endl;
  return s;
}

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_2D_H_
