/*
 * @file volumetric_mesh_wireframe_render_task.h 
 * @Basic wireframe render task of volumetric mesh
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

template <typename Scalar> class CubicMeshWireframeRenderTask;
template <typename Scalar> class QuadMeshWireframeRenderTask;
template <typename Scalar> class TetMeshWireframeRenderTask;
template <typename Scalar> class TriMeshWireframeRenderTask;

template <typename Scalar, int Dim>
class VolumetricMeshWireframeRenderTask:public RenderTaskBase
{
public:
    explicit VolumetricMeshWireframeRenderTask(std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util);
    ~VolumetricMeshWireframeRenderTask() = default;

    //disable copy
    VolumetricMeshWireframeRenderTask(const VolumetricMeshWireframeRenderTask &) = delete;
    VolumetricMeshWireframeRenderTask & operator = (const VolumetricMeshWireframeRenderTask &) = delete;

    void setUniformColor(const Color4f & color);
    void setElementColors(const std::vector<Color4f> & colors);

private:
    void initShader();
    void initWireframeRenderTask();

    void renderTaskImpl() override;

private:
    std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util_;

    std::shared_ptr<CubicMeshWireframeRenderTask<Scalar>> cubic_mesh_wireframe_render_task_;
    std::shared_ptr<QuadMeshWireframeRenderTask<Scalar>> quad_mesh_wireframe_render_task_;
    std::shared_ptr<TetMeshWireframeRenderTask<Scalar>> tet_mesh_wireframe_render_task_;
    std::shared_ptr<TriMeshWireframeRenderTask<Scalar>> tri_mesh_wireframe_render_task_;
};
    
}//end of namespace Physika