/*
* @file vector_Nd.cpp 
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

#include <cmath>
#include <limits>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector_Nd.h"

namespace Physika{

template <typename Scalar>
VectorND<Scalar>::VectorND()
{
    allocMemory(0);
}

template <typename Scalar>
VectorND<Scalar>::VectorND(int dim)
{
    allocMemory(dim);
}

template <typename Scalar>
VectorND<Scalar>::VectorND(int dim, Scalar value)
{
    allocMemory(dim);
    for(int i = 0; i < dim; ++i)
	(*this)[i] = value;
}

template <typename Scalar>
VectorND<Scalar>::VectorND(const VectorND<Scalar> &vec2)
{
    allocMemory(vec2.dims());
    *this = vec2;
}

template <typename Scalar>
void VectorND<Scalar>::allocMemory(int dims)
{
    PHYSIKA_ASSERT(dims>=0);
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    ptr_eigen_vector_Nx_ = new Eigen::Matrix<Scalar,Eigen::Dynamic,1>(dims);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    data_ = new Scalar[dims];
    dims_ = dims;
#endif
}

template <typename Scalar>
VectorND<Scalar>::~VectorND()
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    delete ptr_eigen_vector_Nx_;
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    delete[] data_;
#endif
}

template <typename Scalar>
int VectorND<Scalar>::dims() const
{
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return (*ptr_eigen_vector_Nx_).rows();
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return dims_;
#endif
}

template <typename Scalar>
void VectorND<Scalar>::resize(int new_dim)
{
    PHYSIKA_ASSERT(new_dim>=0);
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    (*ptr_eigen_vector_Nx_).resize(new_dim);
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    if(data_)
	delete[] data_;
    allocMemory(new_dim);
#endif
}

template <typename Scalar>
Scalar& VectorND<Scalar>::operator[] (int idx)
{
    PHYSIKA_ASSERT(idx>=0&&idx<(*this).dims());
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return (*ptr_eigen_vector_Nx_)[idx];
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_[idx];
#endif
}

template <typename Scalar>
const Scalar& VectorND<Scalar>::operator[] (int idx) const
{
    PHYSIKA_ASSERT(idx>=0&&idx<(*this).dims());
#ifdef PHYSIKA_USE_EIGEN_VECTOR
    return (*ptr_eigen_vector_Nx_)[idx];
#elif defined(PHYSIKA_USE_BUILT_IN_VECTOR)
    return data_[idx];
#endif
}

template <typename Scalar>
VectorND<Scalar> VectorND<Scalar>::operator+ (const VectorND<Scalar> &vec2) const
{
    int dim1 = (*this).dims();
    int dim2 = vec2.dims();
    PHYSIKA_ASSERT(dim1==dim2);
    VectorND<Scalar> result(dim1);
    for(int i = 0; i < dim1; ++i)
	result[i] = (*this)[i] + vec2[i];
    return result;
}

template <typename Scalar>
VectorND<Scalar>& VectorND<Scalar>::operator+= (const VectorND<Scalar> &vec2)
{
    int dim1 = (*this).dims();
    int dim2 = vec2.dims();
    PHYSIKA_ASSERT(dim1==dim2);
    for(int i = 0; i < dim1; ++i)
	(*this)[i] = (*this)[i] + vec2[i];
    return *this;
}

template <typename Scalar>
VectorND<Scalar> VectorND<Scalar>::operator- (const VectorND<Scalar> &vec2) const
{
    int dim1 = (*this).dims();
    int dim2 = vec2.dims();
    PHYSIKA_ASSERT(dim1==dim2);
    VectorND<Scalar> result(dim1);
    for(int i = 0; i < dim1; ++i)
	result[i] = (*this)[i] - vec2[i];
    return result;
}

template <typename Scalar>
VectorND<Scalar>& VectorND<Scalar>::operator-= (const VectorND<Scalar> &vec2)
{
    int dim1 = (*this).dims();
    int dim2 = vec2.dims();
    PHYSIKA_ASSERT(dim1==dim2);
    for(int i = 0; i < dim1; ++i)
	(*this)[i] = (*this)[i] - vec2[i];
    return *this;
} 

template <typename Scalar>
VectorND<Scalar>& VectorND<Scalar>::operator= (const VectorND<Scalar> &vec2)
{
    int new_dim = vec2.dims();
    if((*this).dims() != new_dim)
        (*this).resize(new_dim);
    for(int i = 0; i < new_dim; ++i)
	(*this)[i] = vec2[i];
    return *this;
}

template <typename Scalar>
bool VectorND<Scalar>::operator== (const VectorND<Scalar> &vec2) const
{
    int dim1 = (*this).dims();
    int dim2 = vec2.dims();
    if(dim1 != dim2)
	return false;
    for(int i = 0; i <= dim1; ++i)
	if((*this)[i] != vec2[i])
	    return false;
    return true;
}

template <typename Scalar>
VectorND<Scalar> VectorND<Scalar>::operator* (Scalar scale) const
{
    int dim = (*this).dims();
    VectorND<Scalar> result(dim);
    for(int i = 0; i < dim; ++i)
	result[i] = (*this)[i] * scale;
    return result;
}

template <typename Scalar>
VectorND<Scalar>& VectorND<Scalar>::operator*= (Scalar scale)
{
    int dim = (*this).dims();
    for(int i = 0; i < dim; ++i)
	(*this)[i] = (*this)[i] * scale;
    return *this;
}

template <typename Scalar>
VectorND<Scalar> VectorND<Scalar>::operator/ (Scalar scale) const
{
    PHYSIKA_ASSERT(fabs(scale)>std::numeric_limits<Scalar>::epsilon());
    int dim = (*this).dims();
    VectorND<Scalar> result(dim);
    for(int i = 0; i < dim; ++i)
	result[i] = (*this)[i] / scale;
    return result;
}

template <typename Scalar>
VectorND<Scalar>& VectorND<Scalar>::operator/= (Scalar scale)
{
    PHYSIKA_ASSERT(fabs(scale)>std::numeric_limits<Scalar>::epsilon());
    int dim = (*this).dims();
    for(int i = 0; i < dim; ++i)
	(*this)[i] = (*this)[i] / scale;
    return *this;
}

template <typename Scalar>
Scalar VectorND<Scalar>::norm() const
{
    Scalar result = 0.0;
    int dim = (*this).dims();
    for(int i = 0; i < dim; ++i)
	result += (*this)[i]*(*this)[i];
    result = sqrt(result);
    return result;
}

template <typename Scalar>
VectorND<Scalar>& VectorND<Scalar>::normalize()
{
    Scalar norm = (*this).norm();
    bool nonzero_norm = norm > std::numeric_limits<Scalar>::epsilon();
    if(nonzero_norm)
    {
	int dim = (*this).dims();
	for(int i = 0; i < dim; ++i)
	    (*this)[i] = (*this)[i] / norm;
    }
    return *this;
}

template <typename Scalar>
VectorND<Scalar> VectorND<Scalar>::operator - (void ) const
{
    int dim = (*this).dims();
    VectorND<Scalar> result(dim);
    for(int i = 0; i < dim; ++i)
	result[i] = - (*this)[i];
    return result;
}

template <typename Scalar>
Scalar VectorND<Scalar>::dot(const VectorND<Scalar> &vec2) const
{
    int dim1 = (*this).dims();
    int dim2 = vec2.dims();
    PHYSIKA_ASSERT(dim1 == dim2);
    Scalar result = 0.0;
    for(int i = 0; i < dim1; ++i)
	result += (*this)[i]*vec2[i];
    return result;
}

//explicit instantiation
template class VectorND<float>;
template class VectorND<double>;

}//end of namespace Physika
