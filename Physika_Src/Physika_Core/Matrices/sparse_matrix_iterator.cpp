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
#include "Physika_Core/Matrices/sparse_matrix_iterator.h"

namespace Physika{

#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
template <typename Scalar>
SparseMatrixIterator<Scalar>::SparseMatrixIterator(SparseMatrix<Scalar> & mat, unsigned int i)
{
	first_ele_ = mat.line_index_[i];
	last_ele_ = mat.line_index_[i + 1];
	ptr_matrix_ = &mat;
}
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
template <typename Scalar>
SparseMatrixIterator<Scalar>::SparseMatrixIterator(SparseMatrix<Scalar> & mat, unsigned int i):it(*(mat.ptr_eigen_sparse_matrix_), i)
{
}
#endif

template <typename Scalar>
SparseMatrixIterator<Scalar>& SparseMatrixIterator<Scalar>::operator++ ()
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	first_ele_++;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	++it;
#endif
	return *this;
}

template <typename Scalar>
unsigned int SparseMatrixIterator<Scalar>::row() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return (ptr_matrix_->elements_[first_ele_]).row();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return it.row();
#endif
}

template <typename Scalar>
unsigned int SparseMatrixIterator<Scalar>::col() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return (ptr_matrix_->elements_[first_ele_]).col();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return it.col();
#endif
}

template <typename Scalar>
Scalar SparseMatrixIterator<Scalar>::value() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return (ptr_matrix_->elements_[first_ele_]).value();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return it.value();
#endif       
}

template <typename Scalar>
SparseMatrixIterator<Scalar>::operator bool() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
	return first_ele_ != last_ele_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	return (bool)(it);
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


