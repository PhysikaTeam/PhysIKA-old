/*
 * @file square_matrix.h 
 * @brief This abstract class is intended to provide a uniform interface for Matrix2x2 and Matrix3x3.
 *        Matrix2x2 and Matrix3x3 are implemented using template partial specialization of this class.
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

#ifndef PHYSIKA_CORE_MATRICES_SQUARE_MATRIX_H_
#define PHYSIKA_CORE_MATRICES_SQUARE_MATRIX_H_

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/matrix_base.h"

namespace Physika{

template <typename Scalar, int Dim>
class SquareMatrix: public MatrixBase
{
public:
    SquareMatrix(){}
    ~SquareMatrix(){}
    virtual unsigned int rows() const;
    virtual unsigned int cols() const;
private:
    void compileTimeCheck()
    {
        //SquareMatrix<Scalar,Dim> is only defined for 1D&&2D&&3D&&4D with element type of integers and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT(Dim==1||Dim==2||Dim==3||Dim==4,"SquareMatrix<Scalar,Dim> are only defined for Dim==1,2,3,4");
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "SquareMatrix<Scalar,Dim> are only defined for integer types and floating-point types.");
    }
}; 

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_SQUARE_MATRIX_H_
