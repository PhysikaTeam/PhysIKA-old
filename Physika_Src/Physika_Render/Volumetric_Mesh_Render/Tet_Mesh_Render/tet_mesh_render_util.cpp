/*
 * @file tet_mesh_render_util.cpp 
 * @Basic class TetMeshRenderUtil
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


#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Render/Tetrahedron_Render/tetrahedron_render_util.h"

#include "tet_mesh_render_util.h"

namespace Physika{

template <typename Scalar>
TetMeshRenderUtil<Scalar>::TetMeshRenderUtil(TetMesh<Scalar> * mesh, bool auto_compute_normal)
    :VolumetricMeshRenderUtilBase(mesh), mesh_(mesh),
    tet_render_util_(std::make_shared<TetrahedronRenderUtil>())
{
    this->initTetrahedronRenderUtil(auto_compute_normal);
}

template <typename Scalar>
const VolumetricMesh<Scalar, 3> * TetMeshRenderUtil<Scalar>::mesh() const
{
    return mesh_;
}

template <typename Scalar>
void TetMeshRenderUtil<Scalar>::setMesh(VolumetricMesh<Scalar, 3> * mesh, bool auto_compute_normal)
{
    this->mesh_ = static_cast<TetMesh<Scalar> *>(mesh);
    if (this->mesh_ == nullptr)
        throw PhysikaException("error invalid mesh pointer!");

    //call base set mesh
    this->VolumetricMeshRenderUtilBase::setMesh(mesh, auto_compute_normal);

    this->initTetrahedronRenderUtil(auto_compute_normal);
}

template <typename Scalar>
unsigned int TetMeshRenderUtil<Scalar>::tetrahedronNum() const
{
    return tet_render_util_->tetrahedronNum();
}

template <typename Scalar>
unsigned int TetMeshRenderUtil<Scalar>::eleNum() const
{
    return this->tetrahedronNum();
}

template <typename Scalar>
std::shared_ptr<TetrahedronRenderUtil> TetMeshRenderUtil<Scalar>::getInnerTetrahedronRenderUtil()
{
    return tet_render_util_;
}

template <typename Scalar>
void TetMeshRenderUtil<Scalar>::bindTetMeshVAO()
{
    tet_render_util_->bindTetrahedronVAO();
}

template <typename Scalar>
void TetMeshRenderUtil<Scalar>::unbindTetMeshVAO()
{
    tet_render_util_->unbindTetrahedronVAO();
}

template <typename Scalar>
void TetMeshRenderUtil<Scalar>::initTetrahedronRenderUtil(bool auto_compute_normal)
{
    unsigned int point_num = mesh_->vertNum();

    std::vector<Vector<Scalar, 3>> pos_vec(point_num);
    for (unsigned int i = 0; i < point_num; ++i)
        pos_vec[i] = mesh_->vertPos(i);

    unsigned int ele_num = mesh_->eleNum();

    std::vector<unsigned int> indices_vec(4 * ele_num);
    for(unsigned int i = 0; i < ele_num; ++i)
    {
        indices_vec[4 * i] = mesh_->eleVertIndex(i, 0);
        indices_vec[4 * i + 1] = mesh_->eleVertIndex(i, 1);
        indices_vec[4 * i + 2] = mesh_->eleVertIndex(i, 2);
        indices_vec[4 * i + 3] = mesh_->eleVertIndex(i, 3);
    }

    tet_render_util_->setTetrahedrons(pos_vec, indices_vec, auto_compute_normal);
}

//explicit instantiation 
template class TetMeshRenderUtil<float>;
template class TetMeshRenderUtil<double>;

    
}//end of namespace Physika