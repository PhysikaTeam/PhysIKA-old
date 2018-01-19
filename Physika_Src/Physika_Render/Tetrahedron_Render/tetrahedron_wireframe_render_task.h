/*
 * @file tetrahedron_wireframe_render_task.h 
 * @Basic wireframe render task of tetrahedron
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

#include <memory>

#include "Physika_Render/Triangle_Render/triangle_wireframe_render_task.h"


namespace Physika{

class TetrahedronRenderUtil;

//need further consideration for inheritance from TriangleWireframeRenderTask

class TetrahedronWireframeRenderTask: public TriangleWireframeRenderTask
{
public:
    explicit TetrahedronWireframeRenderTask(std::shared_ptr<TetrahedronRenderUtil> render_util);
    ~TetrahedronWireframeRenderTask();

    //disable copy
    TetrahedronWireframeRenderTask(const TetrahedronWireframeRenderTask &) = delete;
    TetrahedronWireframeRenderTask & operator = (const TetrahedronWireframeRenderTask &) = delete;

    void setTetColors(const std::vector<Color4f> & colors);
    void setElementColors(const std::vector<Color4f> & colors) override;

private:
    std::shared_ptr<TetrahedronRenderUtil> render_util_;
};
    
}//end of namespace Physika