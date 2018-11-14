/*
 * @file tri_mesh_point_render_task.cpp
 * @Basic point vector render task of tri mesh
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
#include "tri_mesh_render_util.h"
#include "tri_mesh_point_vector_render_task.h"

namespace Physika{

template <typename Scalar>
TriMeshPointVectorRenderTask<Scalar>::TriMeshPointVectorRenderTask(std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util)
    :PointVectorRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(render_util)
{
    
}

template <typename Scalar>
void TriMeshPointVectorRenderTask<Scalar>::setPointVectorAsBoundaryNormalVector()
{
    //Note: we assume all vertices/faces/elements of TriMesh are boundary.

    const TriMesh<Scalar> * mesh = static_cast<const TriMesh<Scalar> *>(render_util_->mesh());

    std::vector<Vector<Scalar, 3>> normals(render_util_->pointNum());
    for(unsigned int ele_id = 0; ele_id < mesh->eleNum(); ++ele_id)
    {
        unsigned int v0 = mesh->eleVertIndex(ele_id, 0);
        unsigned int v1 = mesh->eleVertIndex(ele_id, 1);
        unsigned int v2 = mesh->eleVertIndex(ele_id, 2);

        Vector<Scalar, 2> temp_p0 = mesh->eleVertPos(ele_id, 0);
        Vector<Scalar, 2> temp_p1 = mesh->eleVertPos(ele_id, 1);
        Vector<Scalar, 2> temp_p2 = mesh->eleVertPos(ele_id, 2);

        Vector<Scalar, 3> p0 = { temp_p0[0], temp_p0[1], 0 };
        Vector<Scalar, 3> p1 = { temp_p1[0], temp_p1[1], 0 };
        Vector<Scalar, 3> p2 = { temp_p2[0], temp_p2[1], 0 };

        //face one
        Vector<Scalar, 3> normal = ((p1 - p0).cross(p2 - p0)).normalize();
        normals[v0] += normal;
        normals[v1] += normal;
        normals[v2] += normal;
    }

    //normalize
    for(unsigned int i = 0; i < normals.size(); ++i)
        normals[i].normalize();

    this->setPointVectors(normals);
}

//explicit instantiation 
template class TriMeshPointVectorRenderTask<float>;
template class TriMeshPointVectorRenderTask<double>;
    
}//end of namespace Physika