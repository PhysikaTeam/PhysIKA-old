/*
 * @file sparse_matrix.h 
 * @brief Definition of sparse matrix, size of the matrix is dynamic.
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

#ifndef PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
#define PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/misc_constants.h"
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

template <typename Scalar, int Rows, int Cols,int StoreMajor = 0>
class SparseMatrix: public MatrixBase
{
 public:
 SparseMatrix();
 ~SparseMatrix();
 inline int rows()const;
 inline int cols()const;
 protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
