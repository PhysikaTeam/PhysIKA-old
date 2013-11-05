/*
 * @file render_base.h 
 * @Basic render, all other render calss inerit from this class.
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

#ifndef PHYSIKA_RENDER_RENDER_BASE_RENDER_BASE_H_
#define PHYSIKA_RENDER_RENDER_BASE_RENDER_BASE_H_

namespace Physika{

class RenderBase
{
public:
    RenderBase();
    virtual ~RenderBase() {};


    virtual void render() {};

protected:
};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_RENDER_BASE_RENDER_BASE_H_
