/*
 * @file face.h 
 * @brief face of 3d surface mesh
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_GEOMETRY_BOUNDARY_MESHES_FACE_H_
#define PHYSIKA_GEOMETRY_BOUNDARY_MESHES_FACE_H_

#include <vector>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Boundary_Meshes/vertex.h"

namespace Physika{

using BoundaryMeshInternal::Vertex;

namespace SurfaceMeshInternal{

template <typename Scalar>
class Face
{
public:
    Face();
    ~Face();
    explicit Face(const std::vector<Vertex<Scalar> > &vertices);
    Face(const std::vector<Vertex<Scalar> > &vertices, const Vector<Scalar,3> &face_normal);
    unsigned int numVertices() const;
    const Vertex<Scalar>& vertex(unsigned int vert_idx) const;
    Vertex<Scalar>& vertex(unsigned int vert_idx);
    const Vertex<Scalar>* vertexPtr(unsigned int vert_idx) const;
    Vertex<Scalar>* vertexPtr(unsigned int vert_idx);
    void setFaceNormal(const Vector<Scalar,3> &face_normal);
    Vector<Scalar,3> faceNormal() const;
    bool hasFaceNormal() const;
    void addVertex(const Vertex<Scalar> &vertex);
    void reverseVertices(); //reverse the order of vertices
    void printVertices() const; //print indices of the vertices
protected:
    std::vector<Vertex<Scalar> > vertices_;
    Vector<Scalar,3> normal_;
    bool has_normal_;
};

} //end of namespace SurfaceMeshInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BOUNDARY_MESHES_FACE_H_
