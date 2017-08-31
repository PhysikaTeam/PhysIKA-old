/*
* @file cuda_array.h
* @Brief class CudaArray, Array allocated in GPU
* @author Wei Chen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/

#include "Physika_Core/Cuda_Array/cuda_array.h"

namespace Physika{

template class CudaArray<float>;
template class CudaArray<double>;

}//end of namespace Physika