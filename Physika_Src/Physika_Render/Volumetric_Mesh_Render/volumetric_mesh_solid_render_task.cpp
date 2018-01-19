/*
 * @file volumetric_mesh_solid_render_task.cpp
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

#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Render/Triangle_Render/triangle_solid_render_shader_srcs.h"

#include "Cubic_Mesh_Render/cubic_mesh_render_util.h"
#include "Quad_Mesh_Render/quad_mesh_render_util.h"
#include "Tet_Mesh_Render/tet_mesh_render_util.h"
#include "Tri_Mesh_Render/tri_mesh_render_util.h"

#include "Cubic_Mesh_Render/cubic_mesh_solid_render_task.h"
#include "Quad_Mesh_Render/quad_mesh_solid_render_task.h"
#include "Tet_Mesh_Render/tet_mesh_solid_render_task.h"
#include "Tri_Mesh_Render/tri_mesh_solid_render_task.h"

#include "volumetric_mesh_render_util.h"
#include "volumetric_mesh_solid_render_task.h"

namespace Physika{

template <typename Scalar, int Dim>
Physika::VolumetricMeshSolidRenderTask<Scalar, Dim>::VolumetricMeshSolidRenderTask(std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util)
    :render_util_(render_util)
{
    this->initShader();
    this->initSolidRenderTask();
}

template <typename Scalar, int Dim>
void VolumetricMeshSolidRenderTask<Scalar, Dim>::initShader()
{
    shader_.createFromCStyleString(triangle_solid_render_vertex_shader, triangle_solid_render_frag_shader);
}

template <typename Scalar, int Dim>
void VolumetricMeshSolidRenderTask<Scalar, Dim>::initSolidRenderTask()
{
    VolumetricMesh<Scalar, Dim> * mesh = const_cast<VolumetricMesh<Scalar, Dim> *>(render_util_->mesh());
    auto render_util_base = render_util_->getInnerVolumetricMeshRenderUtilBase();

    switch (mesh->elementType())
    {
    case VolumetricMeshInternal::CUBIC:
    {
        auto cubic_mesh_render_util = std::dynamic_pointer_cast<CubicMeshRenderUtil<Scalar>>(render_util_base);

        PHYSIKA_ASSERT(cubic_mesh_render_util != nullptr);
        cubic_mesh_solid_render_task_ = std::make_shared<CubicMeshSolidRenderTask<Scalar>>(cubic_mesh_render_util);
        break;
    }


    case VolumetricMeshInternal::QUAD:
    {
        auto quad_mesh_render_util = std::dynamic_pointer_cast<QuadMeshRenderUtil<Scalar>>(render_util_base);

        PHYSIKA_ASSERT(quad_mesh_render_util != nullptr);
        quad_mesh_solid_render_task_ = std::make_shared<QuadMeshSolidRenderTask<Scalar>>(quad_mesh_render_util);
        break;
    }

    case VolumetricMeshInternal::TET:
    {
        auto tet_mesh_render_util = std::dynamic_pointer_cast<TetMeshRenderUtil<Scalar>>(render_util_base);

        PHYSIKA_ASSERT(tet_mesh_render_util != nullptr);
        tet_mesh_solid_render_task_ = std::make_shared<TetMeshSolidRenderTask<Scalar>>(tet_mesh_render_util);
        break;
    }

    case VolumetricMeshInternal::TRI:
    {
        auto tri_mesh_render_util = std::dynamic_pointer_cast<TriMeshRenderUtil<Scalar>>(render_util_base);

        PHYSIKA_ASSERT(tri_mesh_render_util != nullptr);
        tri_mesh_solid_render_task_ = std::make_shared<TriMeshSolidRenderTask<Scalar>>(tri_mesh_render_util);
        break;
    }

    default:
        throw PhysikaException("error: unknown element type!");
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshSolidRenderTask<Scalar, Dim>::renderTaskImpl()
{
    PHYSIKA_ASSERT(cubic_mesh_solid_render_task_ || quad_mesh_solid_render_task_ || tet_mesh_solid_render_task_ || tri_mesh_solid_render_task_);

    if (cubic_mesh_solid_render_task_)
        cubic_mesh_solid_render_task_->renderTaskImpl();
    if (quad_mesh_solid_render_task_)
        quad_mesh_solid_render_task_->renderTaskImpl();
    if (tet_mesh_solid_render_task_)
        tet_mesh_solid_render_task_->renderTaskImpl();
    if (tri_mesh_solid_render_task_)
        tri_mesh_solid_render_task_->renderTaskImpl();
}

//explicit instantiation
template class VolumetricMeshSolidRenderTask<float, 2>;
template class VolumetricMeshSolidRenderTask<float, 3>;
template class VolumetricMeshSolidRenderTask<double, 2>;
template class VolumetricMeshSolidRenderTask<double, 3>;

}//end of namespace Physika


