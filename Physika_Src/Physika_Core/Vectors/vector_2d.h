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

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_2D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_2D_H_

#include <iostream>
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Vectors/vector.h"

namespace Physika{

/*
 * Vector<Scalar,2> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar,2>: public VectorBase
{
public:
    Vector();
    Vector(Scalar x, Scalar y);
    explicit Vector(Scalar);
    Vector(const Vector<Scalar,2>&);
    ~Vector();
    inline int dims() const{return 2;}
    Scalar& operator[] (int);
    const Scalar& operator[] (int) const;
    Vector<Scalar,2> operator+ (const Vector<Scalar,2> &) const;
    Vector<Scalar,2>& operator+= (const Vector<Scalar,2> &);
    Vector<Scalar,2> operator- (const Vector<Scalar,2> &) const;
    Vector<Scalar,2>& operator-= (const Vector<Scalar,2> &);
    Vector<Scalar,2>& operator= (const Vector<Scalar,2> &);
    bool operator== (const Vector<Scalar,2> &) const;
    bool operator!= (const Vector<Scalar,2> &) const;
    Vector<Scalar,2> operator* (Scalar) const;
    Vector<Scalar,2>& operator*= (Scalar);
    Vector<Scalar,2> operator/ (Scalar) const;
    Vector<Scalar,2>& operator/= (Scalar);
    Scalar norm() const;
    Vector<Scalar,2>& normalize();
    Scalar cross(const Vector<Scalar,2> &)const;
    Vector<Scalar,2> operator - (void) const;
    Scalar dot(const Vector<Scalar,2>&) const;

protected:
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    Eigen::Matrix<Scalar,2,1> eigen_vector_2x_;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    Scalar data_[2];
#endif

};

//overriding << for vector2D
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Vector<Scalar,2> &vec)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
        s<<"("<<static_cast<int>(vec[0])<<", "<<static_cast<int>(vec[1])<<")";
    else
        s<<"("<<vec[0]<<", "<<vec[1]<<")";
    return s;
}

//make * operator commutative
template <typename S, typename T>
Vector<T,2> operator *(S scale, const Vector<T,2> &vec)
{
    return vec * scale;
}

//convenient typedefs
typedef Vector<float,2> Vector2f;
typedef Vector<double,2> Vector2d;
typedef Vector<int,2> Vector2i;

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_2D_H_
