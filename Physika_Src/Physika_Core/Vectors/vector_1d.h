/*
 * @file vector_1d.h 
 * @brief 1d vector.
 * @author Fei Zhu
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
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector.h"

namespace Physika{

template <typename Scalar, int Dim> class SquareMatrix;

/*
 * Vector<Scalar,1> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar,1>: public VectorBase
{
public:
    Vector();
    explicit Vector(Scalar);
    Vector(const Vector<Scalar,1>&);
    ~Vector();
    inline unsigned int dims() const{return 1;}
    Scalar& operator[] (unsigned int);
    const Scalar& operator[] (unsigned int) const;
    Vector<Scalar,1> operator+ (const Vector<Scalar,1> &) const;
    Vector<Scalar,1>& operator+= (const Vector<Scalar,1> &);
    Vector<Scalar,1> operator- (const Vector<Scalar,1> &) const;
    Vector<Scalar,1>& operator-= (const Vector<Scalar,1> &);
    Vector<Scalar,1>& operator= (const Vector<Scalar,1> &);
    bool operator== (const Vector<Scalar,1> &) const;
    bool operator!= (const Vector<Scalar,1> &) const;

    Vector<Scalar,1> operator+ (Scalar) const;
    Vector<Scalar,1> operator- (Scalar) const;
    Vector<Scalar,1> operator* (Scalar) const;
    Vector<Scalar,1> operator/ (Scalar) const;

    Vector<Scalar,1>& operator+= (Scalar);
    Vector<Scalar,1>& operator-= (Scalar);
    Vector<Scalar,1>& operator*= (Scalar);
    Vector<Scalar,1>& operator/= (Scalar);

    Scalar norm() const;
    Scalar normSquared() const;
    Vector<Scalar,1>& normalize();
    Scalar cross(const Vector<Scalar,1> &)const;
    Vector<Scalar,1> operator - (void) const;
    Scalar dot(const Vector<Scalar,1>&) const;
    SquareMatrix<Scalar,1> outerProduct(const Vector<Scalar,1>&) const;

protected:
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    Eigen::Matrix<Scalar,1,1> eigen_vector_1x_;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    Scalar data_;
#endif
private:
    void compileTimeCheck()//dummy method for compile time check
    {
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "Vector<Scalar,1> are only defined for integer types and floating-point types.");
    }
};

//overriding << for vector1D
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const Vector<Scalar,1> &vec)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
        s<<"("<<static_cast<int>(vec[0])<<")";
    else
        s<<"("<<vec[0]<<")";
    return s;
}

//make * operator commutative
template <typename S, typename T>
inline Vector<T,1> operator *(S scale, const Vector<T,1> &vec)
{
    return vec * scale;
}

//convenient typedefs
typedef Vector<float,1> Vector1f;
typedef Vector<double,1> Vector1d;
typedef Vector<int,1> Vector1i;

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_1D_H_
