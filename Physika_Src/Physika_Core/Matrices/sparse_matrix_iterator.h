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
        SparseMatrixIterator(SparseMatrix<Scalar> & mat, unsigned int i) :it(*(mat.ptr_eigen_sparse_matrix_), i)
        {
            //it = Eigen::SparseMatrix<Scalar>::InnerIterator (*(mat.ptr_eigen_sparse_matrix_), i);
        }
        SparseMatrixIterator<Scalar>& operator++ ()
        {
            ++it;
            return *this;
        }
        unsigned int row()
        {
            return it.row();
        }
        unsigned int col()
        {
            return it.col();
        }
        Scalar value()
        {
            return it.value();
        }
        operator bool ()
        {
            return it;
        }
    protected:
        typename Eigen::SparseMatrix<Scalar>::InnerIterator it;
    };

}  //end of namespace Physika

//implementation

#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_ITERATOR_H_
