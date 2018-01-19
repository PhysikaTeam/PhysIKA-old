/*
 * @file color_bar_render_task.cpp
 * @Brief render task of color bar.
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

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/ColorBar/color_bar.h"


#include "color_bar_render_task.h"

namespace Physika{

template <typename Scalar>
ColorBarRenderTask<Scalar>::ColorBarRenderTask(const ColorBar<Scalar> & color_bar)
    :color_bar_(color_bar), color_bar_render_(&color_bar_)
{
    //color bar is rendered with fixed opengl pipe, we need disable bind shader
    this->disableBindShader();
}

template <typename Scalar>
void ColorBarRenderTask<Scalar>::setColorBar(const ColorBar<Scalar> & color_bar)
{
    color_bar_ = color_bar;
    color_bar_render_.setColorBar(&color_bar_);
}

template <typename Scalar>
const ColorBar<Scalar> & ColorBarRenderTask<Scalar>::colorBar() const
{
    return color_bar_;
}

template <typename Scalar>
void ColorBarRenderTask<Scalar>::renderTaskImpl()
{
    glVerify(glPushAttrib(GL_ALL_ATTRIB_BITS));
    color_bar_render_.render();
    glVerify(glPopAttrib());
}

//explicit instantiation
template class ColorBarRenderTask<float>;
template class ColorBarRenderTask<double>;
    
}//end of namespace Physika