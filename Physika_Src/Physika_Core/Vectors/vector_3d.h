/*
* @file vector_3d.h 
* @brief 3d vector.
* @author Sheng Yang, Fei Zhu, Wei Chen
* 
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
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

#include "Physika_Core/Utilities/cuda_utilities.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector.h"

namespace Physika{

template <typename Scalar, int Dim> class SquareMatrix;

/*
 * Vector<Scalar,3> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar,3>
{
public:
    CPU_GPU_FUNC_DECL Vector();
    CPU_GPU_FUNC_DECL explicit Vector(Scalar);
    CPU_GPU_FUNC_DECL Vector(Scalar x, Scalar y, Scalar z);
    CPU_GPU_FUNC_DECL Vector(const Vector<Scalar,3>&) = default;
    CPU_GPU_FUNC_DECL ~Vector() = default;

    CPU_GPU_FUNC_DECL static unsigned int dims(){return 3;}

    CPU_GPU_FUNC_DECL Scalar& operator[] (unsigned int);
    CPU_GPU_FUNC_DECL const Scalar& operator[] (unsigned int) const;

    CPU_GPU_FUNC_DECL const Vector<Scalar,3> operator+ (const Vector<Scalar,3> &) const;
    CPU_GPU_FUNC_DECL Vector<Scalar,3>& operator+= (const Vector<Scalar,3> &);
    CPU_GPU_FUNC_DECL const Vector<Scalar,3> operator- (const Vector<Scalar,3> &) const;
    CPU_GPU_FUNC_DECL Vector<Scalar,3>& operator-= (const Vector<Scalar,3> &);

    CPU_GPU_FUNC_DECL Vector<Scalar,3>& operator= (const Vector<Scalar,3> &) = default;

    CPU_GPU_FUNC_DECL bool operator== (const Vector<Scalar,3> &) const;
    CPU_GPU_FUNC_DECL bool operator!= (const Vector<Scalar,3> &) const;

    CPU_GPU_FUNC_DECL const Vector<Scalar,3> operator* (Scalar) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar,3> operator- (Scalar) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar,3> operator+ (Scalar) const;
    CPU_GPU_FUNC_DECL const Vector<Scalar,3> operator/ (Scalar) const;

    CPU_GPU_FUNC_DECL Vector<Scalar,3>& operator+= (Scalar);
    CPU_GPU_FUNC_DECL Vector<Scalar,3>& operator-= (Scalar);
    CPU_GPU_FUNC_DECL Vector<Scalar,3>& operator*= (Scalar);
    CPU_GPU_FUNC_DECL Vector<Scalar,3>& operator/= (Scalar);

    CPU_GPU_FUNC_DECL const Vector<Scalar, 3> operator - (void) const;

    CPU_GPU_FUNC_DECL Scalar norm() const;
    CPU_GPU_FUNC_DECL Scalar normSquared() const;
    CPU_GPU_FUNC_DECL Vector<Scalar,3>& normalize();
    CPU_GPU_FUNC_DECL Vector<Scalar,3> cross(const Vector<Scalar,3> &) const;
    CPU_GPU_FUNC_DECL Scalar dot(const Vector<Scalar,3>&) const;
    CPU_GPU_FUNC_DECL const SquareMatrix<Scalar,3> outerProduct(const Vector<Scalar,3>&) const;
    
protected:

    glm::tvec3<Scalar> data_; //default: zero vector

private:
    void compileTimeCheck()//dummy method for compile time check
    {
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value || is_floating_point<Scalar>::value),
                              "Vector<Scalar,3> are only defined for integer types and  floating-point types.");
    }

};

//overriding << for vector3D
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const Vector<Scalar,3> &vec)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
        s<<"("<<static_cast<int>(vec[0])<<", "<<static_cast<int>(vec[1])<<", "<<static_cast<int>(vec[2])<<")";
    else
        s<<"("<<vec[0]<<", "<<vec[1]<<", "<<vec[2]<<")";
    return s;
}

//make * operator commutative
template <typename S, typename T>
CPU_GPU_FUNC_DECL const Vector<T,3> operator *(S scale, const Vector<T,3> &vec)
{
    return vec * scale;
}


//convenient typedefs 
typedef Vector<float,3> Vector3f;
typedef Vector<double,3> Vector3d;
typedef Vector<int,3> Vector3i;

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_3D_H_
