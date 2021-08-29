/*
 * @file vector_4d.h 
 * @brief 4d vector.
 * @author Liyou Xu, Fei Zhu, Wei Chen
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_4D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_4D_H_

#include <iostream>
#include <glm/vec4.hpp>
#include "vector_base.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class SquareMatrix;

/*
 * Vector<Scalar,4> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar, 4>
{
public:
    typedef Scalar VarType;

    COMM_FUNC Vector();
    COMM_FUNC explicit Vector(Scalar);
    COMM_FUNC Vector(Scalar x, Scalar y, Scalar z, Scalar w);
    COMM_FUNC Vector(const Vector<Scalar, 4>&);
    COMM_FUNC ~Vector();

    COMM_FUNC static const int dims()
    {
        return ( const int )4;
    }

    COMM_FUNC Scalar& operator[](unsigned int);
    COMM_FUNC const Scalar& operator[](unsigned int) const;

    COMM_FUNC const Vector<Scalar, 4> operator+(const Vector<Scalar, 4>&) const;
    COMM_FUNC Vector<Scalar, 4>& operator+=(const Vector<Scalar, 4>&);
    COMM_FUNC const Vector<Scalar, 4> operator-(const Vector<Scalar, 4>&) const;
    COMM_FUNC Vector<Scalar, 4>& operator-=(const Vector<Scalar, 4>&);
    COMM_FUNC const Vector<Scalar, 4> operator*(const Vector<Scalar, 4>&) const;
    COMM_FUNC Vector<Scalar, 4>& operator*=(const Vector<Scalar, 4>&);
    COMM_FUNC const Vector<Scalar, 4> operator/(const Vector<Scalar, 4>&) const;
    COMM_FUNC Vector<Scalar, 4>& operator/=(const Vector<Scalar, 4>&);

    COMM_FUNC Vector<Scalar, 4>& operator=(const Vector<Scalar, 4>&);

    COMM_FUNC bool operator==(const Vector<Scalar, 4>&) const;
    COMM_FUNC bool operator!=(const Vector<Scalar, 4>&) const;

    COMM_FUNC const Vector<Scalar, 4> operator+(Scalar) const;
    COMM_FUNC const Vector<Scalar, 4> operator-(Scalar) const;
    COMM_FUNC const Vector<Scalar, 4> operator*(Scalar) const;
    COMM_FUNC const Vector<Scalar, 4> operator/(Scalar) const;

    COMM_FUNC Vector<Scalar, 4>& operator+=(Scalar);
    COMM_FUNC Vector<Scalar, 4>& operator-=(Scalar);
    COMM_FUNC Vector<Scalar, 4>& operator*=(Scalar);
    COMM_FUNC Vector<Scalar, 4>& operator/=(Scalar);

    COMM_FUNC const Vector<Scalar, 4> operator-(void) const;

    COMM_FUNC Scalar norm() const;
    COMM_FUNC Scalar normSquared() const;
    COMM_FUNC Vector<Scalar, 4>& normalize();
    COMM_FUNC Scalar             dot(const Vector<Scalar, 4>&) const;
    //    COMM_FUNC const SquareMatrix<Scalar,4> outerProduct(const Vector<Scalar,4>&) const;

    COMM_FUNC Vector<Scalar, 4> minimum(const Vector<Scalar, 4>&) const;
    COMM_FUNC Vector<Scalar, 4> maximum(const Vector<Scalar, 4>&) const;

    COMM_FUNC Scalar* getDataPtr()
    {
        return &data_.x;
    }

public:
    glm::tvec4<Scalar> data_;  //default: zero vector
};

//
// //overriding << for vector2D
// template <typename Scalar>
// inline std::ostream& operator<< (std::ostream &s, const Vector<Scalar,4> &vec)
// {
//     if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
//         s<<"("<<static_cast<int>(vec[0])<<", "<<static_cast<int>(vec[1])<<", "<<static_cast<int>(vec[2])<<", "<<static_cast<int>(vec[3])<<")";
//     else
//         s<<"("<<vec[0]<<", "<<vec[1]<<", "<<vec[2]<<", "<<vec[3]<<")";
//     return s;
// }

//make * operator commutative
// template <typename S, typename T>
// COMM_FUNC  const Vector<T,4> operator *(S scale, const Vector<T,4> &vec)
// {
//     return vec * scale;
// }

template class Vector<float, 4>;
template class Vector<double, 4>;
//convenient typedefs
typedef Vector<float, 4>  Vector4f;
typedef Vector<double, 4> Vector4d;
//typedef Vector<int,4> Vector4i;

}  //end of namespace PhysIKA

#include "vector_4d.inl"
#endif  //PHYSIKA_CORE_VECTORS_VECTOR_4D_H_
