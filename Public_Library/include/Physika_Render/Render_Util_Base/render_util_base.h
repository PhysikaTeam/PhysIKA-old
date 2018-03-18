/*
 * @file render_util_base.h 
 * @Basic RenderUtilBase class
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

#pragma once

#include <string>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/matrix_4x4.h"

namespace Physika{

class RenderUtilBase
{
public:
    virtual ~RenderUtilBase() = default;

    virtual void synchronize() = 0;

    //basic draw operation, need further consideration
    virtual void drawBySolid() = 0;
    virtual void drawByPoint() = 0;
};

}//end of namespace Physika