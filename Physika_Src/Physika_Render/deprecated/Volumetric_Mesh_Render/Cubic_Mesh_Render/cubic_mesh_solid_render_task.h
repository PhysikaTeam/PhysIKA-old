/*
 * @file cubic_mesh_solid_render_task.h
 * @Basic solid render task of cubic mesh
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

#include "Physika_Render/Cube_Render/cube_solid_render_task.h"

namespace Physika{

template <typename Scalar>
class CubicMeshRenderUtil;

template <typename Scalar>
class CubicMeshSolidRenderTask: public CubeSolidRenderTask
{
public:
    explicit CubicMeshSolidRenderTask(std::shared_ptr<CubicMeshRenderUtil<Scalar>> render_util);
    ~CubicMeshSolidRenderTask() = default;

    //disable copy
    CubicMeshSolidRenderTask(const CubicMeshSolidRenderTask &) = delete;
    CubicMeshSolidRenderTask & operator = (const CubicMeshSolidRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshSolidRenderTask
    using CubeSolidRenderTask::renderTaskImpl;

private:
    std::shared_ptr<CubicMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika