/*
 * @file quad_mesh_solid_render_task.h 
 * @Basic solid render task of quad mesh
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

#include "Physika_Render/Quad_Render/quad_solid_render_task.h"

namespace Physika{

template <typename Scalar>
class QuadMeshRenderUtil;

template <typename Scalar>
class QuadMeshSolidRenderTask: public QuadSolidRenderTask
{
public:
    explicit QuadMeshSolidRenderTask(std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util);
    ~QuadMeshSolidRenderTask() = default;

    //disable copy
    QuadMeshSolidRenderTask(const QuadMeshSolidRenderTask &) = delete;
    QuadMeshSolidRenderTask & operator = (const QuadMeshSolidRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshSolidRenderTask
    using QuadSolidRenderTask::renderTaskImpl;

private:
    std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika