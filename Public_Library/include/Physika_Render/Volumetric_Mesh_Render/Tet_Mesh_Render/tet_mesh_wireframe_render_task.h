/*
 * @file tet_mesh_wireframe_render_task.h 
 * @Basic wireframe render task of tet mesh
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

#include "Physika_Render/Tetrahedron_Render/tetrahedron_wireframe_render_task.h"

namespace Physika{

template <typename Scalar>
class TetMeshRenderUtil;

template <typename Scalar>
class TetMeshWireframeRenderTask: public TetrahedronWireframeRenderTask
{
public:
    explicit TetMeshWireframeRenderTask(std::shared_ptr<TetMeshRenderUtil<Scalar>> render_util);
    ~TetMeshWireframeRenderTask() = default;

    //disable copy
    TetMeshWireframeRenderTask(const TetMeshWireframeRenderTask &) = delete;
    TetMeshWireframeRenderTask & operator = (const TetMeshWireframeRenderTask &) = delete;

    //expose renderTaskImpl for VolumetricMeshWireframeRenderTask
    using TetrahedronWireframeRenderTask::renderTaskImpl;

private:
    std::shared_ptr<TetMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika