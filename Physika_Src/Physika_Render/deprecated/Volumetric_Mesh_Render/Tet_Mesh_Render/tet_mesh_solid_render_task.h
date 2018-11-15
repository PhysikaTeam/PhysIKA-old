/*
 * @file tet_mesh_solid_render_task.h 
 * @Basic solid render task of tet mesh
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

#include "Physika_Render/Tetrahedron_Render/tetrahedron_solid_render_task.h"

namespace Physika{

template <typename Scalar>
class TetMeshRenderUtil;

template <typename Scalar>
class TetMeshSolidRenderTask: public TetrahedronSolidRenderTask
{
public:
    explicit TetMeshSolidRenderTask(std::shared_ptr<TetMeshRenderUtil<Scalar>> render_util);
    ~TetMeshSolidRenderTask() = default;

    //disable copy
    TetMeshSolidRenderTask(const TetMeshSolidRenderTask &) = delete;
    TetMeshSolidRenderTask & operator = (const TetMeshSolidRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshSolidRenderTask
    using TetrahedronSolidRenderTask::renderTaskImpl;

private:
    std::shared_ptr<TetMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika