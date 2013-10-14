/*
 * @file vector_base.h 
 * @brief Base class of vectors, all vectors inherite from this class.
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

#ifndef PHYSIKA_CORE_VECTORS_VECTOR_BASE_H_
#define PHSYIKA_CORE_VECTORS_VECTOR_BASE_H_

namespace Physika{

template <typename Scalar, int Dims>
class VectorBase
{
 public:
  VectorBase(){};
  ~VectorBase(){};
  virtual int dims() const=0;
 protected:
};

}  //end of namespace Physika

#endif //PHSYIKA_CORE_VECTORS_VECTOR_BASE_H_
