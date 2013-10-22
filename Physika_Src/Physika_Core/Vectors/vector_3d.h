/*
* @file vector_3d.h 
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

#ifndef PHSYIKA_CORE_VECTORS_VECTOR_3D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_3D_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Vectors/vector_base.h"

namespace Physika{

template <typename Scalar>
class Vector3D: public VectorBase
{
public:
	Vector3D();
	Vector3D(Scalar x, Scalar y, Scalar z);
	Vector3D(Scalar);
	~Vector3D();
	inline int dims() const{return 3;}
	Scalar& operator[] (int);
	const Scalar& operator[] (int) const;
	Vector3D<Scalar> operator+ (const Vector3D<Scalar> &) const;
	Vector3D<Scalar>& operator+= (const Vector3D<Scalar> &);
	Vector3D<Scalar> operator- (const Vector3D<Scalar> &) const;
	Vector3D<Scalar>& operator-= (const Vector3D<Scalar> &);
	Vector3D<Scalar>& operator= (const Vector3D<Scalar> &);
	bool operator== (const Vector3D<Scalar> &) const;
	Vector3D<Scalar> operator* (Scalar) const;
	Vector3D<Scalar>& operator*= (Scalar);
	Vector3D<Scalar> operator/ (Scalar) const;
	Vector3D<Scalar>& operator/= (Scalar);
	Scalar norm() const;
	Vector3D<Scalar>& normalize();
	Vector3D<Scalar> cross(const Vector3D<Scalar> &)const;
	Vector3D<Scalar> operator - (void) const;
	Scalar dot(const Vector3D<Scalar>&) const;
	//friend Vector3D<Scalar> operator* (Scalar, const Vector3D<Scalar>&);
	
protected:
#ifdef PHYSIKA_USE_EIGEN_VECTOR
	Eigen::Matrix<Scalar,3,1> eigen_vector_3x_;
#endif

};

//overriding << for vector3D
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Vector3D<Scalar> &vec)
{
	s<<vec[0]<<", "<<vec[1]<<", "<<vec[2]<<std::endl;
	return s;
}

//make * operator commuative
template <typename Scalar>
Vector3D<Scalar> operator *(Scalar scale, Vector3D<Scalar> vec)
{
	return vec * scale;
}

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_3D_H_
