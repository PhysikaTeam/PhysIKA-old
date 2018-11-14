/*
 * @file cubic_mesh_wireframe_render_task.h 
 * @Basic wireframe render task of cubic mesh
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

#include "Physika_Render/Cube_Render/cube_wireframe_render_task.h"

namespace Physika{

template <typename Scalar>
class CubicMeshRenderUtil;

template <typename Scalar>
class CubicMeshWireframeRenderTask: public CubeWireframeRenderTask
{
public:
    explicit CubicMeshWireframeRenderTask(std::shared_ptr<CubicMeshRenderUtil<Scalar>> render_util);
    ~CubicMeshWireframeRenderTask() = default;

    //disable copy
    CubicMeshWireframeRenderTask(const CubicMeshWireframeRenderTask &) = delete;
    CubicMeshWireframeRenderTask & operator = (const CubicMeshWireframeRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshWireframeRenderTask
    using CubeWireframeRenderTask::renderTaskImpl;

private:
    std::shared_ptr<CubicMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika