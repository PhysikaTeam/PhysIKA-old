/*
* @file sparse_matrix_iterator.h
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
#ifndef PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_ITERATOR_H_
#define PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_ITERATOR_H_
#include "Physika_Dependency/Eigen/Eigen"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include <iostream>

namespace Physika{
    
    template <typename Scalar>
    class SparseMatrixIterator
    {
    public:
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
		SparseMatrixIterator(SparseMatrix<Scalar> & mat, unsigned int i)
		{
			first_ele_ = mat.line_index_[i];
			last_ele_ = mat.line_index_[i + 1];
			ptr_matrix_ = &mat;
		}
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
        SparseMatrixIterator(SparseMatrix<Scalar> & mat, unsigned int i) :it(*(mat.ptr_eigen_sparse_matrix_), i)
        {
            //it = Eigen::SparseMatrix<Scalar>::InnerIterator (*(mat.ptr_eigen_sparse_matrix_), i);
        }
#endif
        SparseMatrixIterator<Scalar>& operator++ ()
        {
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
			first_ele_++;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
            ++it;
#endif
            return *this;
        }
        unsigned int row()
        {
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
			return (ptr_matrix_->elements_[first_ele_]).row();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
			return it.row();
#endif
        }
        unsigned int col()
        {
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
			return (ptr_matrix_->elements_[first_ele_]).col();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
			return it.col();
#endif
        }
        Scalar value()
        {
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
			return (ptr_matrix_->elements_[first_ele_]).value();
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
			return it.value();
#endif       
		}
        operator bool ()
        {
            return first_ele_ != last_ele_;
        }
    protected:
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
		unsigned int first_ele_ ,last_ele_;
		SparseMatrix<Scalar> *ptr_matrix_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
        typename Eigen::SparseMatrix<Scalar>::InnerIterator it;
#endif
    };

}  //end of namespace Physika

//implementation

#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_ITERATOR_H_
