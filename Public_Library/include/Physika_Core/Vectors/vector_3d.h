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
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector.h"

namespace Physika{

template <typename Scalar, int Dim> class SquareMatrix;

/*
 * Vector<Scalar,3> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class Vector<Scalar,3>: public VectorBase
{
public:
    Vector();
    explicit Vector(Scalar);
    Vector(Scalar x, Scalar y, Scalar z);
    Vector(const Vector<Scalar,3>&) = default;
    ~Vector() = default;

    inline unsigned int dims() const{return 3;}

    Scalar& operator[] (unsigned int);
    const Scalar& operator[] (unsigned int) const;

    const Vector<Scalar,3> operator+ (const Vector<Scalar,3> &) const;
    Vector<Scalar,3>& operator+= (const Vector<Scalar,3> &);
    const Vector<Scalar,3> operator- (const Vector<Scalar,3> &) const;
    Vector<Scalar,3>& operator-= (const Vector<Scalar,3> &);

    Vector<Scalar,3>& operator= (const Vector<Scalar,3> &) = default;

    bool operator== (const Vector<Scalar,3> &) const;
    bool operator!= (const Vector<Scalar,3> &) const;

    const Vector<Scalar,3> operator* (Scalar) const;
    const Vector<Scalar,3> operator- (Scalar) const;
    const Vector<Scalar,3> operator+ (Scalar) const;
    const Vector<Scalar,3> operator/ (Scalar) const;

    Vector<Scalar,3>& operator+= (Scalar);
    Vector<Scalar,3>& operator-= (Scalar);
    Vector<Scalar,3>& operator*= (Scalar);
    Vector<Scalar,3>& operator/= (Scalar);

    const Vector<Scalar, 3> operator - (void) const;

    Scalar norm() const;
    Scalar normSquared() const;
    Vector<Scalar,3>& normalize();
    Vector<Scalar,3> cross(const Vector<Scalar,3> &) const;
    Scalar dot(const Vector<Scalar,3>&) const;
    const SquareMatrix<Scalar,3> outerProduct(const Vector<Scalar,3>&) const;
    
protected:
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    Eigen::Matrix<Scalar,3,1> eigen_vector_3x_; //default: zero vector
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    Scalar data_[3]; //default: zero vector
#endif
private:
    void compileTimeCheck()//dummy method for compile time check
    {
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "Vector<Scalar,3> are only defined for integer types and floating-point types.");
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
inline const Vector<T,3> operator *(S scale, const Vector<T,3> &vec)
{
    return vec * scale;
}


//convenient typedefs 
typedef Vector<float,3> Vector3f;
typedef Vector<double,3> Vector3d;
typedef Vector<int,3> Vector3i;

} //end of namespace Physika

#endif //PHYSIKA_CORE_VECTORS_VECTOR_3D_H_
