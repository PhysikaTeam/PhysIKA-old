/*
 * @file transform_base.h 
 * @brief Base class of transform, all transform inherite from this class.
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

#ifndef PHYSIKA_CORE_TRANSFORM_TRANSFORM_BASE_H_
#define PHYSIKA_CORE_TRANSFORM_TRANSFORM_BASE_H_

namespace Physika{

class TransformBase
{
public:
    TransformBase(){}
    virtual ~TransformBase(){}
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_TRANSFORM_TRANSFORM_BASE_H_
