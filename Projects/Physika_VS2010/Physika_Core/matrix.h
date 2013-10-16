/*
 * @file matrix.h 
 * @This file contains all matrix-related header files and common used typedefs. 
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_H_

#include "matrix_2x2.h"
#include "matrix_3x3.h"

namespace Physika{
	typedef Matrix2x2<float> Matrix2f;
	typedef Matrix2x2<double> Matrix2d;
//namespace Type
//{
//	
//}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_H_
