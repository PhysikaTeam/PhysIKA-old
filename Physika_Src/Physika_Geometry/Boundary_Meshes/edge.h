/*
 * @file edge.h 
 * @brief Edge of 2d polygon and 3d surface mesh
 * @author Wei Chen, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_GEOMETRY_BOUNDARY_MESHES_EDGE_H_
#define PHYSIKA_GEOMETRY_BOUNDARY_MESHES_EDGE_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Boundary_Meshes/vertex.h"

namespace Physika{

namespace BoundaryMeshInternal{

template <typename Scalar, int Dim>
class Edge
{
public:
    Edge();
    ~Edge();
    Edge(const Vertex<Scalar> &vert1, const Vertex<Scalar> &vert2);
    Edge(const Vertex<Scalar> &vert1, const Vertex<Scalar> &vert2, const Vector<Scalar,Dim> &edge_normal); 

    unsigned int            numVertices() const;
    const Vertex<Scalar>&   vertex(unsigned int vert_idx) const;
    Vertex<Scalar>&         vertex(unsigned int vert_idx);
    const Vertex<Scalar>*   vertexPtr(unsigned int vert_idx) const;
    Vertex<Scalar>*         vertexPtr(unsigned int vert_idx);
    void                    setEdgeNormal(const Vector<Scalar,Dim> &edge_normal);
    Vector<Scalar,Dim> edgeNormal() const;
    bool hasEdgeNormal() const;
    void addVertex(const Vertex<Scalar> &vertex);
    void reverseVertices();     //reverse the order of vertices
    void printVertices() const; //print indices of the vertices
protected:
    std::vector<Vertex<Scalar> > vertices_;
    Vector<Scalar,Dim> normal_;
    bool has_normal_;
};

} //end of namespace PolygonInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BOUNDARY_MESHES_EDGE_H_
