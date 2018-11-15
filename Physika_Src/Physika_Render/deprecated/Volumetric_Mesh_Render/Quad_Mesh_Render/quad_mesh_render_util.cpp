/*
 * @file quad_mesh_render_util.cpp 
 * @Basic class QuadMeshRenderUtil
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


#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Render/Quad_Render/quad_render_util.h"

#include "quad_mesh_render_util.h"

namespace Physika{

template <typename Scalar>
QuadMeshRenderUtil<Scalar>::QuadMeshRenderUtil(QuadMesh<Scalar> * mesh, bool auto_compute_normal)
    :VolumetricMeshRenderUtilBase<Scalar, 2>(mesh), mesh_(mesh),
    quad_render_util_(std::make_shared<QuadRenderUtil>())
{
    this->initQuadRenderUtil(auto_compute_normal);
}

template <typename Scalar>
const VolumetricMesh<Scalar, 2> * QuadMeshRenderUtil<Scalar>::mesh() const
{
    return mesh_;
}

template <typename Scalar>
void QuadMeshRenderUtil<Scalar>::setMesh(VolumetricMesh<Scalar, 2> * mesh, bool auto_compute_normal)
{
    this->mesh_ = static_cast<QuadMesh<Scalar> *>(mesh);
    if (this->mesh_ == nullptr)
        throw PhysikaException("error invalid mesh pointer!");

    //call base set mesh
    this->VolumetricMeshRenderUtilBase<Scalar, 2>::setMesh(mesh, auto_compute_normal);

    this->initQuadRenderUtil(auto_compute_normal);
}

template <typename Scalar>
unsigned int QuadMeshRenderUtil<Scalar>::quadNum() const
{
    return quad_render_util_->quadNum();
}

template <typename Scalar>
unsigned int QuadMeshRenderUtil<Scalar>::eleNum() const
{
    return this->quadNum();
}

template <typename Scalar>
std::shared_ptr<QuadRenderUtil> QuadMeshRenderUtil<Scalar>::getInnerQuadRenderUtil()
{
    return quad_render_util_;
}

template <typename Scalar>
void QuadMeshRenderUtil<Scalar>::bindQuadMeshLineVAO()
{
    quad_render_util_->bindQuadLineVAO();
}

template <typename Scalar>
void QuadMeshRenderUtil<Scalar>::unbindQuadMeshLineVAO()
{
    quad_render_util_->unbindQuadLineVAO();
}

template <typename Scalar>
void QuadMeshRenderUtil<Scalar>::bindQuadMeshVAO()
{
    quad_render_util_->bindQuadVAO();
}

template <typename Scalar>
void QuadMeshRenderUtil<Scalar>::unbindQuadMeshVAO()
{
    quad_render_util_->unbindQuadVAO();
}

template <typename Scalar>
void QuadMeshRenderUtil<Scalar>::initQuadRenderUtil(bool auto_compute_normal)
{
    unsigned int point_num = mesh_->vertNum();

    std::vector<Vector<Scalar, 3>> pos_vec(point_num);
    for (unsigned int i = 0; i < point_num; ++i)
    {
        Vector<Scalar, 2> vert_pos = mesh_->vertPos(i);
        pos_vec[i] = {vert_pos[0], vert_pos[1], 0};
    }
        

    unsigned int ele_num = mesh_->eleNum();

    std::vector<unsigned int> indices_vec(4 * ele_num);
    for(unsigned int i = 0; i < ele_num; ++i)
    {
        indices_vec[4 * i] = mesh_->eleVertIndex(i, 0);
        indices_vec[4 * i + 1] = mesh_->eleVertIndex(i, 1);
        indices_vec[4 * i + 2] = mesh_->eleVertIndex(i, 2);
        indices_vec[4 * i + 3] = mesh_->eleVertIndex(i, 3);
    }

    quad_render_util_->setQuads(pos_vec, indices_vec, auto_compute_normal);
}

//explicit instantiation 
template class QuadMeshRenderUtil<float>;
template class QuadMeshRenderUtil<double>;

    
}//end of namespace Physika