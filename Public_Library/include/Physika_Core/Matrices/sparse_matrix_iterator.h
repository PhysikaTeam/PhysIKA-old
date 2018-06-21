/*
* @file sparse_matrix_iterator.h
* @brief  iterator of SparseMatrix class.
* @author Liyou Xu, Fei Zhu
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/
#ifndef PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_ITERATOR_H_
#define PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_ITERATOR_H_

#include <vector>
#include <iostream>
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Matrices/sparse_matrix_internal.h"

namespace Physika{

template <typename Scalar> class SparseMatrix;
    
template <typename Scalar>
class SparseMatrixIterator
{
public:
    explicit SparseMatrixIterator(SparseMatrix<Scalar> & mat);
    SparseMatrixIterator<Scalar>& operator++();
    unsigned int row() const;
    unsigned int col() const;
    Scalar value() const;
    operator bool () const;
protected:
    SparseMatrix<Scalar> *ptr_matrix_;
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    typename std::vector<SparseMatrixInternal::Trituple<Scalar> >::iterator ele_it_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    unsigned int outer_idx_;
    typename Eigen::SparseMatrix<Scalar>::InnerIterator it_;
#endif
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_ITERATOR_H_
