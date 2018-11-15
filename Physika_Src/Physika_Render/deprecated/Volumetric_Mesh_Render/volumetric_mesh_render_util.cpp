/*
 * @file volumetric_mesh_render_util.h 
 * @Basic class VolumetricMeshRenderUtil
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
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"

#include "Cubic_Mesh_Render/cubic_mesh_render_util.h"
#include "Quad_Mesh_Render/quad_mesh_render_util.h"
#include "Tet_Mesh_Render/tet_mesh_render_util.h"
#include "Tri_Mesh_Render/tri_mesh_render_util.h"


#include "volumetric_mesh_render_util.h"

namespace Physika{

template <typename Scalar, int Dim>
VolumetricMeshRenderUtil<Scalar, Dim>::VolumetricMeshRenderUtil(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal)
{
    this->setRenderUtil(mesh, auto_compute_normal);
}

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar, Dim> * VolumetricMeshRenderUtil<Scalar, Dim>::mesh() const
{
    return render_util_->mesh();
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtil<Scalar, Dim>::setMesh(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal)
{
    this->setRenderUtil(mesh, auto_compute_normal);
}

template <typename Scalar, int Dim>
std::shared_ptr<VolumetricMeshRenderUtilBase<Scalar, Dim>> VolumetricMeshRenderUtil<Scalar, Dim>::getInnerVolumetricMeshRenderUtilBase()
{
    return render_util_;
}

template <typename Scalar, int Dim>
std::shared_ptr<PointRenderUtil> VolumetricMeshRenderUtil<Scalar, Dim>::getInnerPointRenderUtil()
{
    return render_util_->getInnerPointRenderUtil();
}

template <typename Scalar, int Dim>
unsigned int VolumetricMeshRenderUtil<Scalar, Dim>::pointNum() const
{
    return render_util_->pointNum();
}

template <typename Scalar, int Dim>
unsigned int VolumetricMeshRenderUtil<Scalar, Dim>::eleNum() const
{
    return render_util_->eleNum();
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtil<Scalar, Dim>::setRenderUtil(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal)
{
    this->setRenderUtil(mesh, auto_compute_normal, DimensionTrait<Dim>());
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtil<Scalar, Dim>::setRenderUtil(VolumetricMesh<Scalar, 2> * mesh, bool auto_compute_normal, DimensionTrait<2>)
{
    switch (mesh->elementType())
    {
    case VolumetricMeshInternal::QUAD:
        {
            QuadMesh<Scalar> * quad_mesh = static_cast<QuadMesh<Scalar> *>(mesh);
            auto quad_mesh_render_util = std::make_shared<QuadMeshRenderUtil<Scalar>>(quad_mesh, auto_compute_normal);

            PHYSIKA_ASSERT(quad_mesh != nullptr);
            PHYSIKA_ASSERT(quad_mesh_render_util != nullptr);

            render_util_ = std::dynamic_pointer_cast<VolumetricMeshRenderUtilBase<Scalar, Dim>>(quad_mesh_render_util);
            break;
        }
    case VolumetricMeshInternal::TRI:
        {
            TriMesh<Scalar> * tri_mesh = static_cast<TriMesh<Scalar> *>(mesh);
            auto tri_mesh_render_util = std::make_shared<TriMeshRenderUtil<Scalar>>(tri_mesh, auto_compute_normal);

            PHYSIKA_ASSERT(tri_mesh != nullptr);
            PHYSIKA_ASSERT(tri_mesh_render_util != nullptr);

            render_util_ = std::dynamic_pointer_cast<VolumetricMeshRenderUtilBase<Scalar, Dim>>(tri_mesh_render_util);
            break;
        }

    default:
        throw PhysikaException("error: unknown element type!");
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtil<Scalar, Dim>::setRenderUtil(VolumetricMesh<Scalar, 3> * mesh, bool auto_compute_normal, DimensionTrait<3>)
{
    switch (mesh->elementType())
    {
    case VolumetricMeshInternal::CUBIC:
        {
            CubicMesh<Scalar> * cubic_mesh = static_cast<CubicMesh<Scalar> *>(mesh);
            auto cubic_mesh_render_util = std::make_shared<CubicMeshRenderUtil<Scalar>>(cubic_mesh, auto_compute_normal);

            PHYSIKA_ASSERT(cubic_mesh != nullptr);
            PHYSIKA_ASSERT(cubic_mesh_render_util != nullptr);

            render_util_ = std::dynamic_pointer_cast<VolumetricMeshRenderUtilBase<Scalar, Dim>>(cubic_mesh_render_util);
            break;
        }
    case VolumetricMeshInternal::TET:
        {
            TetMesh<Scalar> * tet_mesh = static_cast<TetMesh<Scalar> *>(mesh);
            auto tet_mesh_render_util = std::make_shared<TetMeshRenderUtil<Scalar>>(tet_mesh, auto_compute_normal);

            PHYSIKA_ASSERT(tet_mesh != nullptr);
            PHYSIKA_ASSERT(tet_mesh_render_util != nullptr);

            render_util_ = std::dynamic_pointer_cast<VolumetricMeshRenderUtilBase<Scalar, Dim>>(tet_mesh_render_util);
            break;
        }
        
    default:
        throw PhysikaException("error: unknown element type!");
    }
}

//explicit instantiations
template class VolumetricMeshRenderUtil<float, 2>;
template class VolumetricMeshRenderUtil<float, 3>;
template class VolumetricMeshRenderUtil<double, 2>;
template class VolumetricMeshRenderUtil<double, 3>;

    
}//end of namespace Physika