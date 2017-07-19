/*
 * @file color_bar_render.h 
 * @Brief render of color bar.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_RENDER_COLORBAR_RENDER_COLOR_BAR_RENDER_H_
#define PHYSIKA_RENDER_COLORBAR_RENDER_COLOR_BAR_RENDER_H_

#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Render/ColorBar/color_bar.h"

namespace Physika
{

template <typename Scalar>
class ColorBarRender: public RenderBase
{
public:
    ColorBarRender();
    ColorBarRender(ColorBar<Scalar> * color_bar);
    ~ColorBarRender();

    const ColorBar<Scalar> * colorBar() const;
    void setColorBar(ColorBar<Scalar> * color_bar);
    void render();

protected:

    void drawBox(const Vector<Scalar, 3> & start_pos, Scalar dx, Scalar dy, Scalar dz, const Color<Scalar> & color);

protected:
    ColorBar<Scalar> * color_bar_;
};

} // end of namespace Physika

#endif // PHYSIKA_RENDER_COLORBAR_RENDER_COLOR_BAR_RENDER_H_