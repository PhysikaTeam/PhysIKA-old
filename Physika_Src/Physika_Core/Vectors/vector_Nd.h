/*
* @file vector_Nd.h 
* @brief Arbitrary dimension vector, dimension could be changed at runtime.
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

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_ND_H_
#define PHYSIKA_CORE_VECTORS_VECTOR_ND_H_

#include <iostream>
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Vectors/vector_base.h"

namespace Physika{

template <typename Scalar> class MatrixMxN;

/*
 * VectorND<Scalar> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class VectorND: public VectorBase
{
public:
    VectorND();//empty vector, dim = 0
    explicit VectorND(unsigned int dim);//vector with given dimension
    VectorND(unsigned int dim, Scalar value);//vector with given dimension initialized with one value
    VectorND(const VectorND<Scalar>&);
    ~VectorND();
    unsigned int dims() const;
    void resize(unsigned int new_dim);
    Scalar& operator[] (unsigned int);
    const Scalar& operator[] (unsigned int) const;
    VectorND<Scalar> operator+ (const VectorND<Scalar> &) const;
    VectorND<Scalar>& operator+= (const VectorND<Scalar> &);
    VectorND<Scalar> operator- (const VectorND<Scalar> &) const;
    VectorND<Scalar>& operator-= (const VectorND<Scalar> &);
    VectorND<Scalar>& operator= (const VectorND<Scalar> &);
    bool operator== (const VectorND<Scalar> &) const;
    bool operator!= (const VectorND<Scalar> &) const;

    VectorND<Scalar> operator+ (Scalar) const;
    VectorND<Scalar> operator- (Scalar) const;
    VectorND<Scalar> operator* (Scalar) const;
    VectorND<Scalar> operator/ (Scalar) const;

    VectorND<Scalar>& operator+= (Scalar);
    VectorND<Scalar>& operator-= (Scalar);
    VectorND<Scalar>& operator*= (Scalar);
    VectorND<Scalar>& operator/= (Scalar);

    Scalar norm() const;
    Scalar normSquared() const;
    VectorND<Scalar>& normalize();
    VectorND<Scalar> operator - (void) const;
    Scalar dot(const VectorND<Scalar>&) const;
    MatrixMxN<Scalar> outerProduct(const VectorND<Scalar>&) const;

protected:
    void allocMemory(unsigned int dims);
protected:
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> *ptr_eigen_vector_Nx_;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    Scalar *data_;
    unsigned int dims_;
#endif
};

//overriding << for vectorND
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const VectorND<Scalar> &vec)
{
    unsigned int dim = vec.dims();
    s<<"(";
    for(unsigned int i = 0; i < dim-1; ++i)
    {
        if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
            s<<static_cast<unsigned int>(vec[i])<<", ";
        else
            s<<vec[i]<<", ";
    }
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
        s<<static_cast<unsigned int>(vec[dim-1])<<")";
    else
        s<<vec[dim-1]<<")";
    return s;
}

//make * operator commutative
template <typename S, typename T>
inline VectorND<T> operator *(S scale, const VectorND<T> &vec)
{
    return vec * scale;
}

}//end of namespace Physika

#endif//PHYSIKA_CORE_VECTORS_VECTOR_ND_H_
