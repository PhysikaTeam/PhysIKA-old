/*
 * @file color_bar_render_task.h 
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

#pragma once

#include "Physika_Render/Render_Task_Base/render_task_base.h"
#include "Physika_Render/ColorBar/color_bar.h"

#include "color_bar_render.h"

namespace Physika
{

//Note: we use adapter design pattern here.

template <typename Scalar> class ColorBarRender;

template <typename Scalar>
class ColorBarRenderTask: public RenderTaskBase
{
public:
    ColorBarRenderTask() = default;
    explicit ColorBarRenderTask(const ColorBar<Scalar> & color_bar);

    RenderTaskType type() const override; //return RenderTaskType::SCREEN_BASED_RENDER_TASK

    void setColorBar(const ColorBar<Scalar> & color_bar);
    const ColorBar<Scalar> & colorBar() const;

protected:
    void renderTaskImpl() final;

private:
    ColorBar<Scalar> color_bar_;
    ColorBarRender<Scalar> color_bar_render_;
};

} // end of namespace Physika
