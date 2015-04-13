/*
* @file sparse_matrix_iterator.cpp
* @brief  iterator of SparseMatrix class.
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

#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Matrices/sparse_matrix_iterator.h"

namespace Physika{

#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
template <typename Scalar>
SparseMatrixIterator<Scalar>::SparseMatrixIterator(SparseMatrix<Scalar> & mat)
    :ptr_matrix_(&mat),ele_it_(mat.elements_.begin())
{   
}
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
template <typename Scalar>
SparseMatrixIterator<Scalar>::SparseMatrixIterator(SparseMatrix<Scalar> & mat)
    :ptr_matrix_ (&mat),outer_idx_(0),it_(*(mat.ptr_eigen_sparse_matrix_), 0)
{
}
#endif

template <typename Scalar>
SparseMatrixIterator<Scalar>& SparseMatrixIterator<Scalar>::operator++ ()
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
        ++ele_it_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	++it_;
    if(!it_ && outer_idx_ < ptr_matrix_->eigen_sparse_matrix_->outerSize())  //next row/col if reached end of this inner loop
    {
        ++outer_idx_;
        it_ = Eigen::SparseMatrix<Scalar>::InnerIterator(*(ptr_matrix_->eigen_sparse_matrix_), outer_idx_);
    }
#endif
	return *this;
}

template <typename Scalar>
unsigned int SparseMatrixIterator<Scalar>::row() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return (*ele_it_).row();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return it_.row();
#endif
}

template <typename Scalar>
unsigned int SparseMatrixIterator<Scalar>::col() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return (*ele_it_).col();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return it_.col();
#endif
}

template <typename Scalar>
Scalar SparseMatrixIterator<Scalar>::value() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return (*ele_it_).value();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return it_.value();
#endif       
}

template <typename Scalar>
SparseMatrixIterator<Scalar>::operator bool() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return ele_it_ != (ptr_matrix_->elements_).end();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return (bool)(it_);
#endif 
}

template class SparseMatrixIterator<unsigned char>;
template class SparseMatrixIterator<unsigned short>;
template class SparseMatrixIterator<unsigned int>;
template class SparseMatrixIterator<unsigned long>;
template class SparseMatrixIterator<unsigned long long>;
template class SparseMatrixIterator<signed char>;
template class SparseMatrixIterator<short>;
template class SparseMatrixIterator<int>;
template class SparseMatrixIterator<long>;
template class SparseMatrixIterator<long long>;
template class SparseMatrixIterator<float>;
template class SparseMatrixIterator<double>;
template class SparseMatrixIterator<long double>;

}
 //end of namespace Physika


