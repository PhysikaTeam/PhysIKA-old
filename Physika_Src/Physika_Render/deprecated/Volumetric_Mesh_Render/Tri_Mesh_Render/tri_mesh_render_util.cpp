/*
 * @file tri_mesh_render_util.cpp 
 * @Basic class TriMeshRenderUtil
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


#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Render/Triangle_Render/triangle_render_util.h"

#include "tri_mesh_render_util.h"

namespace Physika{

template <typename Scalar>
TriMeshRenderUtil<Scalar>::TriMeshRenderUtil(TriMesh<Scalar> * mesh, bool auto_compute_normal)
    :VolumetricMeshRenderUtilBase<Scalar, 2>(mesh), mesh_(mesh),
    tri_render_util_(std::make_shared<TriangleRenderUtil>())
{
    this->initTriangleRenderUtil(auto_compute_normal);
}

template <typename Scalar>
const VolumetricMesh<Scalar, 2> * TriMeshRenderUtil<Scalar>::mesh() const
{
    return mesh_;
}

template <typename Scalar>
void TriMeshRenderUtil<Scalar>::setMesh(VolumetricMesh<Scalar, 2> * mesh, bool auto_compute_normal)
{
    this->mesh_ = static_cast<TriMesh<Scalar> *>(mesh);
    if (this->mesh_ == nullptr)
        throw PhysikaException("error invalid mesh pointer!");

    //call base set mesh
    this->VolumetricMeshRenderUtilBase<Scalar, 2>::setMesh(mesh, auto_compute_normal);

    this->initTriangleRenderUtil(auto_compute_normal);
}

template <typename Scalar>
unsigned int TriMeshRenderUtil<Scalar>::triangleNum() const
{
    return tri_render_util_->triangleNum();
}

template <typename Scalar>
unsigned int TriMeshRenderUtil<Scalar>::eleNum() const
{
    return this->triangleNum();
}

template <typename Scalar>
std::shared_ptr<TriangleRenderUtil> TriMeshRenderUtil<Scalar>::getInnerTriangleRenderUtil()
{
    return tri_render_util_;
}

template <typename Scalar>
void TriMeshRenderUtil<Scalar>::bindTriMeshVAO()
{
    tri_render_util_->bindTriangleVAO();
}

template <typename Scalar>
void TriMeshRenderUtil<Scalar>::unbindTriMeshVAO()
{
    tri_render_util_->unbindTriangleVAO();
}

template <typename Scalar>
void TriMeshRenderUtil<Scalar>::initTriangleRenderUtil(bool auto_compute_normal)
{
    unsigned int point_num = mesh_->vertNum();

    std::vector<Vector<Scalar, 3>> pos_vec(point_num);
    for (unsigned int i = 0; i < point_num; ++i)
    {
        Vector<Scalar, 2> vert_pos = mesh_->vertPos(i);
        pos_vec[i] = {vert_pos[0], vert_pos[1], 0};
    }
        

    unsigned int ele_num = mesh_->eleNum();

    std::vector<unsigned int> indices_vec(3 * ele_num);
    for(unsigned int i = 0; i < ele_num; ++i)
    {
        indices_vec[3 * i] = mesh_->eleVertIndex(i, 0);
        indices_vec[3 * i + 1] = mesh_->eleVertIndex(i, 1);
        indices_vec[3 * i + 2] = mesh_->eleVertIndex(i, 2);
    }

    tri_render_util_->setTriangles(pos_vec, indices_vec, auto_compute_normal);
}

//explicit instantiation 
template class TriMeshRenderUtil<float>;
template class TriMeshRenderUtil<double>;

    
}//end of namespace Physika