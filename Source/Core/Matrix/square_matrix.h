/*
 * @file square_matrix.h 
 * @brief This abstract class is intended to provide a uniform interface for Matrix2x2 and Matrix3x3.
 *        Matrix2x2 and Matrix3x3 are implemented using template partial specialization of this class.
 * @author Fei Zhu
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_SQUARE_MATRIX_H_
#define PHYSIKA_CORE_MATRICES_SQUARE_MATRIX_H_

#include "matrix_base.h"

namespace PhysIKA {

template <typename Scalar, int Dim>
class SquareMatrix : public MatrixBase
{
public:
    SquareMatrix() {}
    ~SquareMatrix() {}
    COMM_FUNC virtual unsigned int rows() const;
    COMM_FUNC virtual unsigned int cols() const;
};

}  //end of namespace PhysIKA

#endif  //PHYSIKA_CORE_MATRICES_SQUARE_MATRIX_H_
