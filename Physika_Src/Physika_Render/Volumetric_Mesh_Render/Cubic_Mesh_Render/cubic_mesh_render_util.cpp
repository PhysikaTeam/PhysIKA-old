/*
 * @file cubic_mesh_render_util.cpp 
 * @Basic class CubicMeshRenderUtil
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


#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Render/Cube_Render/cube_render_util.h"

#include "cubic_mesh_render_util.h"

namespace Physika{

template <typename Scalar>
CubicMeshRenderUtil<Scalar>::CubicMeshRenderUtil(CubicMesh<Scalar> * mesh, bool auto_compute_normal)
    :VolumetricMeshRenderUtilBase(mesh), mesh_(mesh),
    cube_render_util_(std::make_shared<CubeRenderUtil>())
{
    this->initCubeRenderUtil(auto_compute_normal);
}

template <typename Scalar>
const VolumetricMesh<Scalar, 3> * CubicMeshRenderUtil<Scalar>::mesh() const
{
    return mesh_;
}

template <typename Scalar>
void CubicMeshRenderUtil<Scalar>::setMesh(VolumetricMesh<Scalar, 3> * mesh, bool auto_compute_normal)
{
    this->mesh_ = static_cast<CubicMesh<Scalar> *>(mesh);
    if (this->mesh_ == nullptr)
        throw PhysikaException("error: invalid mesh pointer!");

    //call base set mesh
    this->VolumetricMeshRenderUtilBase::setMesh(mesh, auto_compute_normal);

    this->initCubeRenderUtil(auto_compute_normal);
}

template <typename Scalar>
unsigned int CubicMeshRenderUtil<Scalar>::cubeNum() const
{
    return cube_render_util_->cubeNum();
}

template <typename Scalar>
unsigned int CubicMeshRenderUtil<Scalar>::eleNum() const
{
    return this->cubeNum();
}

template <typename Scalar>
std::shared_ptr<CubeRenderUtil> CubicMeshRenderUtil<Scalar>::getInnerCubeRenderUtil()
{
    return cube_render_util_;
}

template <typename Scalar>
void CubicMeshRenderUtil<Scalar>::bindCubicMeshLineVAO()
{
    cube_render_util_->bindCubeLineVAO();
}

template <typename Scalar>
void CubicMeshRenderUtil<Scalar>::unbindCubicMeshLineVAO()
{
    cube_render_util_->unbindCubeLineVAO();
}

template <typename Scalar>
void CubicMeshRenderUtil<Scalar>::bindCubicMeshVAO()
{
    cube_render_util_->bindCubeVAO();
}

template <typename Scalar>
void CubicMeshRenderUtil<Scalar>::unbindCubicMeshVAO()
{
    cube_render_util_->unbindCubeVAO();
}

template <typename Scalar>
void CubicMeshRenderUtil<Scalar>::initCubeRenderUtil(bool auto_compute_normal)
{
    unsigned int point_num = mesh_->vertNum();

    std::vector<Vector<Scalar, 3>> pos_vec(point_num);
    for (unsigned int i = 0; i < point_num; ++i)
    {
        Vector<Scalar, 3> vert_pos = mesh_->vertPos(i);
        pos_vec[i] = {vert_pos[0], vert_pos[1], vert_pos[2]};
    }
        

    unsigned int ele_num = mesh_->eleNum();

    std::vector<unsigned int> indices_vec(8 * ele_num);
    for(unsigned int i = 0; i < ele_num; ++i)
    {
        indices_vec[8 * i] = mesh_->eleVertIndex(i, 0);
        indices_vec[8 * i + 1] = mesh_->eleVertIndex(i, 1);
        indices_vec[8 * i + 2] = mesh_->eleVertIndex(i, 2);
        indices_vec[8 * i + 3] = mesh_->eleVertIndex(i, 3);
        indices_vec[8 * i + 4] = mesh_->eleVertIndex(i, 4);
        indices_vec[8 * i + 5] = mesh_->eleVertIndex(i, 5);
        indices_vec[8 * i + 6] = mesh_->eleVertIndex(i, 6);
        indices_vec[8 * i + 7] = mesh_->eleVertIndex(i, 7);
    }

    cube_render_util_->setCubes(pos_vec, indices_vec, auto_compute_normal);
}

//explicit instantiation 
template class CubicMeshRenderUtil<float>;
template class CubicMeshRenderUtil<double>;

    
}//end of namespace Physika