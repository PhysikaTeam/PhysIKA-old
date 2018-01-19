/*
 * @file quad_mesh_point_render_task.cpp
 * @Basic point vector render task of quad mesh
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
#include "quad_mesh_render_util.h"
#include "quad_mesh_point_vector_render_task.h"

namespace Physika{

template <typename Scalar>
QuadMeshPointVectorRenderTask<Scalar>::QuadMeshPointVectorRenderTask(std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util)
    :PointVectorRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(render_util)
{
    
}

template <typename Scalar>
void QuadMeshPointVectorRenderTask<Scalar>::setPointVectorAsBoundaryNormalVector()
{
    //Note: we assume all vertices/faces/elements of QuadMesh are boundary.

    const QuadMesh<Scalar> * mesh = static_cast<const QuadMesh<Scalar> *>(render_util_->mesh());

    std::vector<Vector<Scalar, 3>> normals(render_util_->pointNum());
    for (unsigned int ele_id = 0; ele_id < mesh->eleNum(); ++ele_id)
    {
        unsigned int v0 = mesh->eleVertIndex(ele_id, 0);
        unsigned int v1 = mesh->eleVertIndex(ele_id, 1);
        unsigned int v2 = mesh->eleVertIndex(ele_id, 2);
        unsigned int v3 = mesh->eleVertIndex(ele_id, 3);

        Vector<Scalar, 2> temp_p0 = mesh->eleVertPos(ele_id, 0);
        Vector<Scalar, 2> temp_p1 = mesh->eleVertPos(ele_id, 1);
        Vector<Scalar, 2> temp_p2 = mesh->eleVertPos(ele_id, 2);
        Vector<Scalar, 2> temp_p3 = mesh->eleVertPos(ele_id, 3);

        Vector<Scalar, 3> p0 = { temp_p0[0], temp_p0[1], 0 };
        Vector<Scalar, 3> p1 = { temp_p1[0], temp_p1[1], 0 };
        Vector<Scalar, 3> p2 = { temp_p2[0], temp_p2[1], 0 };
        Vector<Scalar, 3> p3 = { temp_p3[0], temp_p3[1], 0 };

        //face one
        Vector<Scalar, 3> normal_one = ((p1 - p0).cross(p2 - p0)).normalize();
        normals[v0] += normal_one;
        normals[v1] += normal_one;
        normals[v2] += normal_one;

        //face two
        Vector<Scalar, 3> normal_two = ((p2 - p0).cross(p3 - p0)).normalize();
        normals[v0] += normal_two;
        normals[v2] += normal_two;
        normals[v3] += normal_two;
    }

    //normalize
    for (unsigned int i = 0; i < normals.size(); ++i)
        normals[i].normalize();

    this->setPointVectors(normals);
}

//explicit instantiation 
template class QuadMeshPointVectorRenderTask<float>;
template class QuadMeshPointVectorRenderTask<double>;
    
}//end of namespace Physika