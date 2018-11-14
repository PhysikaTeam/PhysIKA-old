/*
 * @file tri_mesh_solid_render_task.h 
 * @Basic solid render task of tri mesh
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

#include "Physika_Render/Triangle_Render/triangle_solid_render_task.h"

namespace Physika{

template <typename Scalar>
class TriMeshRenderUtil;

template <typename Scalar>
class TriMeshSolidRenderTask: public TriangleSolidRenderTask
{
public:
    explicit TriMeshSolidRenderTask(std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util);
    ~TriMeshSolidRenderTask() = default;

    //disable copy
    TriMeshSolidRenderTask(const TriMeshSolidRenderTask &) = delete;
    TriMeshSolidRenderTask & operator = (const TriMeshSolidRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshSolidRenderTask
    using TriangleSolidRenderTask::renderTaskImpl;

private:
    std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika