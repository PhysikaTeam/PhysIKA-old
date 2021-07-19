/*
 * @file vector_2d.h
 * @brief 2d vector.
 * @author Fei Zhu, Wei Chen
 *
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_2D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_2D_H_

#include <iostream>
#include <glm/vec2.hpp>
#include "vector_base.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class SquareMatrix;

/*
 * Vector<Scalar,2> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar, 2>
{
public:
    typedef Scalar VarType;

    COMM_FUNC Vector();
    COMM_FUNC explicit Vector(Scalar);
    COMM_FUNC Vector(Scalar x, Scalar y);
    COMM_FUNC Vector(const Vector<Scalar, 2>&);
    COMM_FUNC ~Vector();

    COMM_FUNC static const int dims()
    {
        return 2;
    }

    COMM_FUNC Scalar& operator[](unsigned int);
    COMM_FUNC const Scalar& operator[](unsigned int) const;

    COMM_FUNC const Vector<Scalar, 2> operator+(const Vector<Scalar, 2>&) const;
    COMM_FUNC Vector<Scalar, 2>& operator+=(const Vector<Scalar, 2>&);
    COMM_FUNC const Vector<Scalar, 2> operator-(const Vector<Scalar, 2>&) const;
    COMM_FUNC Vector<Scalar, 2>& operator-=(const Vector<Scalar, 2>&);
    COMM_FUNC const Vector<Scalar, 2> operator*(const Vector<Scalar, 2>&) const;
    COMM_FUNC Vector<Scalar, 2>& operator*=(const Vector<Scalar, 2>&);
    COMM_FUNC const Vector<Scalar, 2> operator/(const Vector<Scalar, 2>&) const;
    COMM_FUNC Vector<Scalar, 2>& operator/=(const Vector<Scalar, 2>&);

    COMM_FUNC Vector<Scalar, 2>& operator=(const Vector<Scalar, 2>&);

    COMM_FUNC bool operator==(const Vector<Scalar, 2>&) const;
    COMM_FUNC bool operator!=(const Vector<Scalar, 2>&) const;

    COMM_FUNC const Vector<Scalar, 2> operator*(Scalar) const;
    COMM_FUNC const Vector<Scalar, 2> operator-(Scalar) const;
    COMM_FUNC const Vector<Scalar, 2> operator+(Scalar) const;
    COMM_FUNC const Vector<Scalar, 2> operator/(Scalar) const;

    COMM_FUNC Vector<Scalar, 2>& operator+=(Scalar);
    COMM_FUNC Vector<Scalar, 2>& operator-=(Scalar);
    COMM_FUNC Vector<Scalar, 2>& operator*=(Scalar);
    COMM_FUNC Vector<Scalar, 2>& operator/=(Scalar);

    COMM_FUNC const Vector<Scalar, 2> operator-(void) const;

    COMM_FUNC Scalar norm() const;
    COMM_FUNC Scalar normSquared() const;
    COMM_FUNC Vector<Scalar, 2>& normalize();
    COMM_FUNC Scalar             cross(const Vector<Scalar, 2>&) const;
    COMM_FUNC Scalar             dot(const Vector<Scalar, 2>&) const;
    COMM_FUNC Vector<Scalar, 2> minimum(const Vector<Scalar, 2>&) const;
    COMM_FUNC Vector<Scalar, 2> maximum(const Vector<Scalar, 2>&) const;
    //    COMM_FUNC const SquareMatrix<Scalar,2> outerProduct(const Vector<Scalar,2>&) const;
    COMM_FUNC Scalar* getDataPtr()
    {
        return &data_.x;
    }

public:
    glm::tvec2<Scalar> data_;
};

//overriding << for vector2D
// template <typename Scalar>
// inline std::ostream& operator<< (std::ostream &s, const Vector<Scalar,2> &vec)
// {
//     if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
//         s<<"("<<static_cast<int>(vec[0])<<", "<<static_cast<int>(vec[1])<<")";
//     else
//         s<<"("<<vec[0]<<", "<<vec[1]<<")";
//     return s;
// }

template class Vector<float, 2>;
template class Vector<double, 2>;
//convenient typedefs
typedef Vector<float, 2>  Vector2f;
typedef Vector<double, 2> Vector2d;
//typedef Vector<int,2> Vector2i;

}  //end of namespace PhysIKA

#include "vector_2d.inl"

#endif  //PHYSIKA_CORE_VECTORS_VECTOR_2D_H_
