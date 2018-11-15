/*
 * @file tet_mesh_point_render_task.cpp
 * @Basic point vector render task of tet mesh
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
#include "tet_mesh_render_util.h"
#include "tet_mesh_point_vector_render_task.h"

namespace Physika{

template <typename Scalar>
TetMeshPointVectorRenderTask<Scalar>::TetMeshPointVectorRenderTask(std::shared_ptr<TetMeshRenderUtil<Scalar>> render_util)
    :PointVectorRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(render_util)
{
    
}

template <typename Scalar>
void TetMeshPointVectorRenderTask<Scalar>::setPointVectorAsBoundaryNormalVector()
{
    //need further consideration
    TetMesh<Scalar> * mesh = static_cast<TetMesh<Scalar> *>(
                                                             const_cast<VolumetricMesh<Scalar, 3> *>(render_util_->mesh())
                                                           );

    std::vector<unsigned int> boundary_eles;
    mesh->boundaryElements(boundary_eles);

    std::vector<Vector<Scalar, 3>> normals(render_util_->pointNum());
    for(unsigned int i = 0; i < boundary_eles.size(); ++i)
    {
        unsigned int ele_id = boundary_eles[i];

        unsigned int v0 = mesh->eleVertIndex(ele_id, 0);
        unsigned int v1 = mesh->eleVertIndex(ele_id, 1);
        unsigned int v2 = mesh->eleVertIndex(ele_id, 2);
        unsigned int v3 = mesh->eleVertIndex(ele_id, 3);

        Vector<Scalar, 3> p0 = mesh->eleVertPos(ele_id, 0);
        Vector<Scalar, 3> p1 = mesh->eleVertPos(ele_id, 1);
        Vector<Scalar, 3> p2 = mesh->eleVertPos(ele_id, 2);
        Vector<Scalar, 3> p3 = mesh->eleVertPos(ele_id, 3);

        //face one
        Vector<Scalar, 3> normal_one = ((p2 - p0).cross(p1 - p0)).normalize();
        normals[v0] += normal_one;
        normals[v2] += normal_one;
        normals[v1] += normal_one;

        //face two
        Vector<Scalar, 3> normal_two = ((p1 - p0).cross(p3 - p0)).normalize();
        normals[v0] += normal_two;
        normals[v1] += normal_two;
        normals[v3] += normal_two;

        //face one
        Vector<Scalar, 3> normal_three = ((p2 - p1).cross(p3 - p1)).normalize();
        normals[v1] += normal_three;
        normals[v2] += normal_three;
        normals[v3] += normal_three;

        //face one
        Vector<Scalar, 3> normal_four = ((p3 - p0).cross(p2 - p0)).normalize();
        normals[v0] += normal_four;
        normals[v3] += normal_four;
        normals[v2] += normal_four;
    }

    //normalize or reset
    for(unsigned int i = 0; i < normals.size(); ++i)
    {
        if (mesh->isBoundaryVertex(i) == true)
            normals[i].normalize();
        else
            normals[i] = Vector<Scalar, 3>(0);
    }

    this->setPointVectors(normals);
}

//explicit instantiation 
template class TetMeshPointVectorRenderTask<float>;
template class TetMeshPointVectorRenderTask<double>;
    
}//end of namespace Physika