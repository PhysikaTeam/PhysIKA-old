/*
 * @file volumetric_mesh_render_util_base.cpp 
 * @Basic class VolumetricMeshRenderUtilBase
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

#include <vector>

#include "Physika_Render/Point_Render/point_render_util.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"

#include "volumetric_mesh_render_util_base.h"

namespace Physika{

template <typename Scalar, int Dim>
VolumetricMeshRenderUtilBase<Scalar, Dim>::VolumetricMeshRenderUtilBase(VolumetricMesh<Scalar, Dim> * mesh)
    :mesh_(mesh),
    point_render_util_(std::make_shared<PointRenderUtil>())
{
    this->initPointRenderUtil();
}

template <typename Scalar, int Dim>
VolumetricMeshRenderUtilBase<Scalar, Dim>::~VolumetricMeshRenderUtilBase() = default;

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar, Dim> * VolumetricMeshRenderUtilBase<Scalar, Dim>::mesh() const
{
    return mesh_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtilBase<Scalar, Dim>::setMesh(VolumetricMesh<Scalar, Dim> * mesh, bool auto_compute_normal)
{
    this->mesh_ = mesh;
    this->initPointRenderUtil();
}

template <typename Scalar, int Dim>
unsigned int VolumetricMeshRenderUtilBase<Scalar, Dim>::pointNum() const
{
    return point_render_util_->pointNum();
}

template <typename Scalar, int Dim>
std::shared_ptr<PointRenderUtil> VolumetricMeshRenderUtilBase<Scalar, Dim>::getInnerPointRenderUtil()
{
    return point_render_util_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtilBase<Scalar, Dim>::bindPointVAO()
{
    point_render_util_->bindPointVAO();
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtilBase<Scalar, Dim>::unbindPointVAO()
{
    point_render_util_->unbindPointVAO();
}

template <typename Scalar, int Dim>
void VolumetricMeshRenderUtilBase<Scalar, Dim>::initPointRenderUtil()
{
    unsigned int point_num = mesh_->vertNum();

    std::vector<Vector<Scalar, Dim>> point_vec(point_num);
    for (unsigned int i = 0; i < point_num; ++i)
        point_vec[i] = mesh_->vertPos(i);

    point_render_util_->setPoints(point_vec);
}

//explicit instantiation 
template class VolumetricMeshRenderUtilBase<float, 2>;
template class VolumetricMeshRenderUtilBase<float, 3>;
template class VolumetricMeshRenderUtilBase<double, 2>;
template class VolumetricMeshRenderUtilBase<double, 3>;

    
}//end of namespace Physika