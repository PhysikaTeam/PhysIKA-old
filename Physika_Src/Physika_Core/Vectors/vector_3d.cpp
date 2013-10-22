/*
 * @file vector_3d.cpp 
 * @brief 3d vector.
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cmath>
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

template <typename Scalar>
Vector3D<Scalar>::Vector3D()
{
}

template <typename Scalar>
Vector3D<Scalar>::Vector3D(Scalar x, Scalar y, Scalar z)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
  eigen_vector_3x_(0)=x;
  eigen_vector_3x_(1)=y;
  eigen_vector_3x_(2)=z;
#endif
}

template <typename Scalar>
Vector3D<Scalar>::Vector3D(Scalar x)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
  eigen_vector_3x_(0)=x;
  eigen_vector_3x_(1)=x;
  eigen_vector_3x_(2)=x;
#endif
}

template <typename Scalar>
Vector3D<Scalar>::~Vector3D()
{
}

template <typename Scalar>
Scalar& Vector3D<Scalar>::operator[] (int idx)
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
  return eigen_vector_3x_(idx);
#endif
}

template <typename Scalar>
const Scalar& Vector3D<Scalar>::operator[] (int idx) const
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
  return eigen_vector_3x_(idx);
#endif
}

template <typename Scalar>
Vector3D<Scalar> Vector3D<Scalar>::operator+ (const Vector3D<Scalar> &vec3) const
{
  Scalar result[3];
  for(int i = 0; i < 3; ++i)
    result[i] = (*this)[i] + vec3[i];
  return Vector3D<Scalar>(result[0],result[1],result[2]);
}

template <typename Scalar>
Vector3D<Scalar>& Vector3D<Scalar>::operator+= (const Vector3D<Scalar> &vec3)
{
  for(int i = 0; i < 3; ++i)
    (*this)[i] = (*this)[i] + vec3[i];
  return *this;
}

template <typename Scalar>
Vector3D<Scalar> Vector3D<Scalar>::operator- (const Vector3D<Scalar> &vec3) const
{
Scalar result[3];
  for(int i = 0; i < 3; ++i)
    result[i] = (*this)[i] - vec3[i];
  return Vector3D<Scalar>(result[0],result[1],result[2]);
}

template <typename Scalar>
Vector3D<Scalar>& Vector3D<Scalar>::operator-= (const Vector3D<Scalar> &vec3)
{
for(int i = 0; i < 3; ++i)
    (*this)[i] = (*this)[i] - vec3[i];
  return *this;
}

template <typename Scalar>
Vector3D<Scalar>& Vector3D<Scalar>::operator= (const Vector3D<Scalar> &vec3)
{
  for(int i = 0; i < 3; ++i)
    (*this)[i] = vec3[i];
  return *this;
}

template <typename Scalar>
bool Vector3D<Scalar>::operator== (const Vector3D<Scalar> &vec3) const
{
  for(int i = 0; i < 3; ++i)
    if((*this)[i] != vec3[i])
      return false;
  return true;
}

template <typename Scalar>
Vector3D<Scalar> Vector3D<Scalar>::operator* (Scalar scale) const
{
  Scalar result[3];
  for(int i = 0; i < 3; ++i)
    result[i] = (*this)[i] * scale;
  return Vector3D<Scalar>(result[0],result[1],result[2]);
}

template <typename Scalar>
Vector3D<Scalar>& Vector3D<Scalar>::operator*= (Scalar scale)
{
  for(int i = 0; i < 3; ++i)
    (*this)[i] = (*this)[i] * scale;
  return *this;
}

template <typename Scalar>
Vector3D<Scalar> Vector3D<Scalar>::operator/ (Scalar scale) const
{
  Scalar result[3];
  for(int i = 0; i < 3; ++i)
    result[i] = (*this)[i] / scale;
  return Vector3D<Scalar>(result[0],result[1],result[2]);
}

template <typename Scalar>
Vector3D<Scalar>& Vector3D<Scalar>::operator/= (Scalar scale)
{
  for(int i = 0; i < 3; ++i)
    (*this)[i] = (*this)[i] / scale;
  return *this;
}

template <typename Scalar>
Scalar Vector3D<Scalar>::norm() const
{
  Scalar result = (*this)[0]*(*this)[0] + (*this)[1]*(*this)[1] + (*this)[2]*(*this)[2];
  result = sqrt(result);
  return result;
}

template <typename Scalar>
Vector3D<Scalar>& Vector3D<Scalar>::normalize()
{
  Scalar norm = (*this).norm();
  if(norm)
  {
    for(int i = 0; i < 3; ++i)
      (*this)[i] = (*this)[i] / norm;
  }
  return *this;
}

template <typename Scalar>
Vector3D<Scalar> Vector3D<Scalar>::cross(const Vector3D<Scalar>& vec3) const
{
	return Vector3D<Scalar>((*this)[1]*vec3[2] - (*this)[2]*vec3[1], (*this)[2]*vec3[0] - (*this)[0]*vec3[2], (*this)[0]*vec3[1] - (*this)[1]*vec3[0]); 
}

template <typename Scalar>
Vector3D<Scalar> Vector3D<Scalar>::operator-(void)const
{
	return Vector3D<Scalar>(-(*this)[0],-(*this)[1],-(*this)[2]);
}

template <typename Scalar>
Scalar Vector3D<Scalar>::dot(const Vector3D<Scalar>& vec3) const
{
	return (*this)[0]*vec3[0] + (*this)[1]*vec3[1] + (*this)[2]*vec3[2];
}


//explicit instantiation of template so that it could be compiled into a lib
//template class Vector3D<int>;
template class Vector3D<float>;
template class Vector3D<double>;

} //end of namespace Physika
