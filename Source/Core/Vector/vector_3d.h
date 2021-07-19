/*
* @file vector_3d.h 
* @brief 3d vector.
* @author Sheng Yang, Fei Zhu, Wei Chen
* 
* This file is part of PhysIKA, a versatile physics simulation library.
* Copyright (C) 2013- PhysIKA Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_3D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_3D_H_

#include <iostream>
#include <glm/vec3.hpp>
#include "vector_base.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class SquareMatrix;

/*
 * Vector<Scalar,3> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar, 3>
{
public:
    typedef Scalar VarType;

    COMM_FUNC Vector();
    COMM_FUNC explicit Vector(Scalar);
    COMM_FUNC Vector(Scalar x, Scalar y, Scalar z);
    COMM_FUNC Vector(const Vector<Scalar, 3>&);
    COMM_FUNC ~Vector();

    COMM_FUNC static int dims()
    {
        return 3;
    }

    COMM_FUNC Scalar& operator[](unsigned int);
    COMM_FUNC const Scalar& operator[](unsigned int) const;

    COMM_FUNC const Vector<Scalar, 3> operator+(const Vector<Scalar, 3>&) const;
    COMM_FUNC Vector<Scalar, 3>& operator+=(const Vector<Scalar, 3>&);
    COMM_FUNC const Vector<Scalar, 3> operator-(const Vector<Scalar, 3>&) const;
    COMM_FUNC Vector<Scalar, 3>& operator-=(const Vector<Scalar, 3>&);
    COMM_FUNC const Vector<Scalar, 3> operator*(const Vector<Scalar, 3>&) const;
    COMM_FUNC Vector<Scalar, 3>& operator*=(const Vector<Scalar, 3>&);
    COMM_FUNC const Vector<Scalar, 3> operator/(const Vector<Scalar, 3>&) const;
    COMM_FUNC Vector<Scalar, 3>& operator/=(const Vector<Scalar, 3>&);

    COMM_FUNC Vector<Scalar, 3>& operator=(const Vector<Scalar, 3>&);

    COMM_FUNC bool operator==(const Vector<Scalar, 3>&) const;
    COMM_FUNC bool operator!=(const Vector<Scalar, 3>&) const;

    COMM_FUNC const Vector<Scalar, 3> operator*(Scalar) const;
    COMM_FUNC const Vector<Scalar, 3> operator-(Scalar) const;
    COMM_FUNC const Vector<Scalar, 3> operator+(Scalar) const;
    COMM_FUNC const Vector<Scalar, 3> operator/(Scalar) const;

    COMM_FUNC Vector<Scalar, 3>& operator+=(Scalar);
    COMM_FUNC Vector<Scalar, 3>& operator-=(Scalar);
    COMM_FUNC Vector<Scalar, 3>& operator*=(Scalar);
    COMM_FUNC Vector<Scalar, 3>& operator/=(Scalar);

    COMM_FUNC const Vector<Scalar, 3> operator-(void) const;

    COMM_FUNC Scalar norm() const;
    COMM_FUNC Scalar normSquared() const;
    COMM_FUNC Vector<Scalar, 3>& normalize();
    COMM_FUNC Vector<Scalar, 3> cross(const Vector<Scalar, 3>&) const;
    COMM_FUNC Scalar            dot(const Vector<Scalar, 3>&) const;
    //    COMM_FUNC const SquareMatrix<Scalar,3> outerProduct(const Vector<Scalar,3>&) const;

    COMM_FUNC Vector<Scalar, 3> minimum(const Vector<Scalar, 3>&) const;
    COMM_FUNC Vector<Scalar, 3> maximum(const Vector<Scalar, 3>&) const;

    COMM_FUNC Scalar* getDataPtr()
    {
        return &data_.x;
    }

public:
    glm::tvec3<Scalar> data_;  //default: zero vector
};

template class Vector<float, 3>;
template class Vector<double, 3>;
//convenient typedefs
typedef Vector<float, 3>  Vector3f;
typedef Vector<double, 3> Vector3d;
}  //end of namespace PhysIKA

#include "vector_3d.inl"

#endif  //PHYSIKA_CORE_VECTORS_VECTOR_3D_H_
