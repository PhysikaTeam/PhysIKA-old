/*
 * @file vector_4d.h 
 * @brief 4d vector.
 * @author Liyou Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_4D_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_4D_H_

#include <iostream>
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Vectors/vector.h"

namespace Physika{

/*
 * Vector<Scalar,4> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar,4>: public VectorBase
{
public:
    Vector();
    Vector(Scalar x, Scalar y, Scalar z, Scalar w);
    explicit Vector(Scalar);
    Vector(const Vector<Scalar,4>&);
    ~Vector();
    inline int dims() const{return 4;}
    Scalar& operator[] (int);
    const Scalar& operator[] (int) const;
    Vector<Scalar,4> operator+ (const Vector<Scalar,4> &) const;
    Vector<Scalar,4>& operator+= (const Vector<Scalar,4> &);
    Vector<Scalar,4> operator- (const Vector<Scalar,4> &) const;
    Vector<Scalar,4>& operator-= (const Vector<Scalar,4> &);
    Vector<Scalar,4>& operator= (const Vector<Scalar,4> &);
    bool operator== (const Vector<Scalar,4> &) const;
    bool operator!= (const Vector<Scalar,4> &) const;
    Vector<Scalar,4> operator* (Scalar) const;
    Vector<Scalar,4>& operator*= (Scalar);
    Vector<Scalar,4> operator/ (Scalar) const;
    Vector<Scalar,4>& operator/= (Scalar);
    Scalar norm() const;
    Vector<Scalar,4>& normalize();
    Vector<Scalar,4> operator - (void) const;
    Scalar dot(const Vector<Scalar,4>&) const;

protected:
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    Eigen::Matrix<Scalar,4,1> eigen_vector_4x_;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    Scalar data_[4];
#endif

};

//overriding << for vector2D
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const Vector<Scalar,4> &vec)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
        s<<"("<<static_cast<int>(vec[0])<<", "<<static_cast<int>(vec[1])<<", "<<static_cast<int>(vec[2])<<", "<<static_cast<int>(vec[3])<<")";
    else
        s<<"("<<vec[0]<<", "<<vec[1]<<", "<<vec[2]<<", "<<vec[3]<<")";
    return s;
}

//make * operator commutative
template <typename S, typename T>
Vector<T,4> operator *(S scale, const Vector<T,4> &vec)
{
    return vec * scale;
}

//convenient typedefs
typedef Vector<float,4> Vector4f;
typedef Vector<double,4> Vector4d;
typedef Vector<int,4> Vector4i;

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_4D_H_
