/*
 * @file volumetric_mesh_solid_render_task.h 
 * @Basic solid render task of volumetric mesh
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
#include "Physika_Render/Render_Task_Base/render_task_base.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMeshRenderUtil;

template <typename Scalar> class CubicMeshSolidRenderTask;
template <typename Scalar> class QuadMeshSolidRenderTask;
template <typename Scalar> class TetMeshSolidRenderTask;
template <typename Scalar> class TriMeshSolidRenderTask;

template <typename Scalar, int Dim>
class VolumetricMeshSolidRenderTask:public RenderTaskBase
{
public:
    explicit VolumetricMeshSolidRenderTask(std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util);
    ~VolumetricMeshSolidRenderTask() = default;

    //disable copy
    VolumetricMeshSolidRenderTask(const VolumetricMeshSolidRenderTask &) = delete;
    VolumetricMeshSolidRenderTask & operator = (const VolumetricMeshSolidRenderTask &) = delete;

    void setUnifromColor(const Color4f & color);
    void setElementColors(const std::vector<Color4f> & colors);

private:
    void initShader();
    void initSolidRenderTask();

    void renderTaskImpl() override;

private:
    std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util_;

    std::shared_ptr<CubicMeshSolidRenderTask<Scalar>> cubic_mesh_solid_render_task_;
    std::shared_ptr<QuadMeshSolidRenderTask<Scalar>> quad_mesh_solid_render_task_;
    std::shared_ptr<TetMeshSolidRenderTask<Scalar>> tet_mesh_solid_render_task_;
    std::shared_ptr<TriMeshSolidRenderTask<Scalar>> tri_mesh_solid_render_task_;
};
    
}//end of namespace Physika