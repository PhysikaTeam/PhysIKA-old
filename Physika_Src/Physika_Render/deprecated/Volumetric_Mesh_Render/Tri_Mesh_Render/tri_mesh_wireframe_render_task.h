/*
 * @file tri_mesh_wireframe_render_task.h 
 * @Basic wireframe render task of tri mesh
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

template <typename Scalar>
class TriMeshRenderUtil;

template <typename Scalar>
class TriMeshWireframeRenderTask: public TriangleWireframeRenderTask
{
public:
    explicit TriMeshWireframeRenderTask(std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util);
    ~TriMeshWireframeRenderTask() = default;

    //disable copy
    TriMeshWireframeRenderTask(const TriMeshWireframeRenderTask &) = delete;
    TriMeshWireframeRenderTask & operator = (const TriMeshWireframeRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshWireframeRenderTask
    using TriangleWireframeRenderTask::renderTaskImpl;

private:
    std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika