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
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Vectors/vector.h"

namespace Physika{

template <typename Scalar, int Dim> class SquareMatrix;

/*
 * Vector<Scalar,1> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar,1>
{
public:
    CPU_GPU_FUNC_DECL Vector();
    CPU_GPU_FUNC_DECL explicit Vector(Scalar);
    CPU_GPU_FUNC_DECL Vector(const Vector<Scalar,1>&) = default;
    CPU_GPU_FUNC_DECL ~Vector() = default;

    CPU_GPU_FUNC_DECL static unsigned int dims(){return 1;}

    CPU_GPU_FUNC_DECL Scalar& operator[] (unsigned int);
    CPU_GPU_FUNC_DECL const Scalar& operator[] (unsigned int) const;

    CPU_GPU_FUNC_DECL const Vector<Scalar,1> operator+ (const Vector<Scalar,1> &) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar, 1> operator- (const Vector<Scalar, 1> &) const;

    CPU_GPU_FUNC_DECL Vector<Scalar,1>& operator+= (const Vector<Scalar,1> &);
    CPU_GPU_FUNC_DECL Vector<Scalar,1>& operator-= (const Vector<Scalar,1> &);

    CPU_GPU_FUNC_DECL Vector<Scalar,1>& operator= (const Vector<Scalar,1> &) = default;

    CPU_GPU_FUNC_DECL bool operator== (const Vector<Scalar,1> &) const;
    CPU_GPU_FUNC_DECL bool operator!= (const Vector<Scalar,1> &) const;

    CPU_GPU_FUNC_DECL const Vector<Scalar,1> operator+ (Scalar) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar,1> operator- (Scalar) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar,1> operator* (Scalar) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar,1> operator/ (Scalar) const;

    CPU_GPU_FUNC_DECL Vector<Scalar,1>& operator+= (Scalar);
    CPU_GPU_FUNC_DECL Vector<Scalar,1>& operator-= (Scalar);
    CPU_GPU_FUNC_DECL Vector<Scalar,1>& operator*= (Scalar);
    CPU_GPU_FUNC_DECL Vector<Scalar,1>& operator/= (Scalar);

    CPU_GPU_FUNC_DECL const Vector<Scalar, 1> operator - (void) const;

    CPU_GPU_FUNC_DECL Scalar norm() const;
    CPU_GPU_FUNC_DECL Scalar normSquared() const;
    CPU_GPU_FUNC_DECL Vector<Scalar,1>& normalize();
    CPU_GPU_FUNC_DECL Scalar cross(const Vector<Scalar,1> &)const;
    CPU_GPU_FUNC_DECL Scalar dot(const Vector<Scalar,1>&) const;
    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,1> outerProduct(const Vector<Scalar,1>&) const;

protected:
    Scalar data_; //default: 0

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
CPU_GPU_FUNC_DECL const Vector<T,1> operator *(S scale, const Vector<T,1> &vec)
{
    return vec * scale;
}

//convenient typedefs
typedef Vector<float,1> Vector1f;
typedef Vector<double,1> Vector1d;
typedef Vector<int,1> Vector1i;

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_1D_H_
