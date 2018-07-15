/*
 * @file vector_1d.h 
 * @brief 1d vector.
 * @author Fei Zhu, Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_1D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_1D_H_

#include <iostream>
#include "vector_base.h"

namespace Physika{

template <typename Scalar, int Dim> class SquareMatrix;

/*
 * Vector<Scalar,1> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar,1>
{
public:
	typedef Scalar VarType;

    COMM_FUNC Vector(); //default: 0
    COMM_FUNC explicit Vector(Scalar);
    COMM_FUNC Vector(const Vector<Scalar,1>&) = default;
    COMM_FUNC ~Vector() = default;

    COMM_FUNC static const int dims(){return 1;}

    COMM_FUNC Scalar& operator[] (unsigned int);
    COMM_FUNC const Scalar& operator[] (unsigned int) const;

    COMM_FUNC const Vector<Scalar,1> operator+ (const Vector<Scalar,1> &) const;
    COMM_FUNC const Vector<Scalar, 1> operator- (const Vector<Scalar, 1> &) const;

    COMM_FUNC Vector<Scalar,1>& operator+= (const Vector<Scalar,1> &);
    COMM_FUNC Vector<Scalar,1>& operator-= (const Vector<Scalar,1> &);

    COMM_FUNC Vector<Scalar,1>& operator= (const Vector<Scalar,1> &) = default;

    COMM_FUNC bool operator== (const Vector<Scalar,1> &) const;
    COMM_FUNC bool operator!= (const Vector<Scalar,1> &) const;

    COMM_FUNC const Vector<Scalar,1> operator+ (Scalar) const;
    COMM_FUNC const Vector<Scalar,1> operator- (Scalar) const;
    COMM_FUNC const Vector<Scalar,1> operator* (Scalar) const;
    COMM_FUNC const Vector<Scalar,1> operator/ (Scalar) const;

    COMM_FUNC Vector<Scalar,1>& operator+= (Scalar);
    COMM_FUNC Vector<Scalar,1>& operator-= (Scalar);
    COMM_FUNC Vector<Scalar,1>& operator*= (Scalar);
    COMM_FUNC Vector<Scalar,1>& operator/= (Scalar);

    COMM_FUNC const Vector<Scalar, 1> operator - (void) const;

    COMM_FUNC Scalar norm() const;
    COMM_FUNC Scalar normSquared() const;
    COMM_FUNC Vector<Scalar,1>& normalize();
    COMM_FUNC Scalar cross(const Vector<Scalar,1> &)const;
    COMM_FUNC Scalar dot(const Vector<Scalar,1>&) const;
//    COMM_FUNC const SquareMatrix<Scalar,1> outerProduct(const Vector<Scalar,1>&) const;

protected:
    Scalar data_; //default: 0
};

//overriding << for vector1D
// template <typename Scalar>
// inline std::ostream& operator<< (std::ostream &s, const Vector<Scalar,1> &vec)
// {
//     if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
//         s<<"("<<static_cast<int>(vec[0])<<")";
//     else
//         s<<"("<<vec[0]<<")";
//     return s;
// }

template class Vector<float, 1>;
template class Vector<double, 1>;
//convenient typedefs
typedef Vector<float,1> Vector1f;
typedef Vector<double,1> Vector1d;
//typedef Vector<int,1> Vector1i;

} //end of namespace Physika

#include "vector_1d.inl"

#endif //PHYSIKA_CORE_VECTORS_VECTOR_1D_H_
