/*
 * @file surface_mesh_point_vector_render_task.cpp
 * @Basic render task of surface mesh vertex with custom vector
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

#include <glm/glm.hpp>

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_Geometry/Boundary_Meshes/face_group.h"

#include "surface_mesh_render_util.h"
#include "surface_mesh_normal_render_shader_srcs.h"
#include "surface_mesh_point_vector_render_task.h"

namespace Physika{

using BoundaryMeshInternal::Vertex;
using SurfaceMeshInternal::Face;
using SurfaceMeshInternal::FaceGroup;
using BoundaryMeshInternal::Material;

template <typename Scalar>
SurfaceMeshPointVectorRenderTask<Scalar>::SurfaceMeshPointVectorRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util)
    :PointVectorRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(std::move(render_util))
{

}
template <typename Scalar>
void SurfaceMeshPointVectorRenderTask<Scalar>::setPointVectorsAsNormalVector()
{
    const SurfaceMesh<Scalar> * mesh = render_util_->mesh();
    unsigned int vert_num = mesh->numVertices();

    //resize point vector
    std::vector<Vector3f> point_vectors(vert_num, Vector3f(0.0f));

    //extract normal vector from mesh
    unsigned int group_num = mesh->numGroups();
    for (unsigned int group_idx = 0; group_idx < group_num; ++group_idx)
    {
        const FaceGroup<Scalar> & group_ref = mesh->group(group_idx);
        unsigned int face_num = group_ref.numFaces();

        for (unsigned int face_idx = 0; face_idx < face_num; ++face_idx)
        {
            const Face<Scalar> & face_ref = group_ref.face(face_idx);          
            unsigned int vertex_num = face_ref.numVertices();

            for(unsigned int vertex_idx = 0;  vertex_idx < vertex_num; ++vertex_idx)
            {
                const Vertex<Scalar> & vertex = face_ref.vertex(vertex_idx);
                const Vector<Scalar, 3> & normal = mesh->vertexNormal(vertex);

                unsigned int position_ID = vertex.positionIndex();
                point_vectors[position_ID] += Vector3f(normal[0], normal[1], normal[2]); //add
            }

        }
    }

    //normalize
    for (int i = 0; i < point_vectors.size(); ++i)
        point_vectors[i].normalize();

    this->setPointVectors(point_vectors);
}

//explicit instantiation
template class SurfaceMeshPointVectorRenderTask<float>;
template class SurfaceMeshPointVectorRenderTask<double>;

}//end of namespace Physika