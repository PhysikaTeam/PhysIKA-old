/*
 * @file quad_mesh_wireframe_render_task.h 
 * @Basic wireframe render task of quad mesh
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

#include "Physika_Render/Quad_Render/quad_wireframe_render_task.h"

namespace Physika{

template <typename Scalar>
class QuadMeshRenderUtil;

template <typename Scalar>
class QuadMeshWireframeRenderTask: public QuadWireframeRenderTask
{
public:
    explicit QuadMeshWireframeRenderTask(std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util);
    ~QuadMeshWireframeRenderTask() = default;

    //disable copy
    QuadMeshWireframeRenderTask(const QuadMeshWireframeRenderTask &) = delete;
    QuadMeshWireframeRenderTask & operator = (const QuadMeshWireframeRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshWireframeRenderTask
    using QuadWireframeRenderTask::renderTaskImpl;

private:
    std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika