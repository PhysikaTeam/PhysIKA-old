/*
 * @file plane_render_task.h 
 * @Basic render task of plane
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

#include <vector>

#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Render/Render_Task_Base/render_task_base.h"

namespace Physika {

class PlaneRenderTask: public RenderTaskBase
{
public:
    PlaneRenderTask();
    ~PlaneRenderTask();

    //disable copy
    PlaneRenderTask(const PlaneRenderTask &) = delete;
    PlaneRenderTask & operator = (const PlaneRenderTask &) = delete;

    void addPlane(const Vector4f & plane, float plane_size = 200.0f);

    void enableRenderGrid();
    void disableRenderGrid();
    bool isRenderGrid() const;

private:
    void renderTaskImpl() override;

    void addPlaneVAOAndVBO(const Vector4f & plane, float plane_size);
    void destoryAllVAOAndVBO();

    void getBasisFromNormalVector(const Vector<float, 3> & w, Vector<float, 3> & u, Vector<float, 3> & v);

private:
    std::vector<Vector4f> planes_;
    std::vector<float> plane_sizes_;
    
    bool render_grid_ = true; //grid render

    std::vector<unsigned int> plane_VAOs_;
    std::vector<unsigned int> plane_VBOs_;
    std::vector<unsigned int> plane_vert_nums_;
};

}//end of namespace Physika