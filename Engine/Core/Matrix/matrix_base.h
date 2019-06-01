/*
 * @file matrix_base.h 
 * @brief Base class of Matrix, all Matrix inherite from this class.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_BASE_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_BASE_H_

namespace Physika{

class MatrixBase
{
public:
    MatrixBase(){}
    virtual ~MatrixBase(){}
    virtual unsigned int rows() const=0;
    virtual unsigned int cols() const=0;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_BASE_H_
