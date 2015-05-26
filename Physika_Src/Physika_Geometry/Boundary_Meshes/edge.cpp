/*
 * @file edge.cpp 
 * @brief edge of 2d polygon and 3d surface mesh
 * @author Fei Zhu, Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include <algorithm>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Boundary_Meshes/edge.h"
using std::vector;

namespace Physika{

namespace BoundaryMeshInternal{

template<typename Scalar, int Dim>
Edge<Scalar,Dim>::Edge():has_normal_(false){}

template<typename Scalar, int Dim>
Edge<Scalar,Dim>::~Edge(){}

template <typename Scalar, int Dim>
Edge<Scalar,Dim>::Edge(const Vertex<Scalar> &vert1, const Vertex<Scalar> &vert2)
    :has_normal_(false)
{
    addVertex(vert1);
    addVertex(vert2);
}

template <typename Scalar, int Dim>
Edge<Scalar,Dim>::Edge(const Vertex<Scalar> &vert1, const Vertex<Scalar> &vert2, const Vector<Scalar,Dim> &edge_normal)
    :normal_(edge_normal),has_normal_(true)
{
    addVertex(vert1);
    addVertex(vert2);
}

template <typename Scalar, int Dim>
unsigned int Edge<Scalar,Dim>::numVertices() const
{
    return vertices_.size();
}

template <typename Scalar, int Dim>
const Vertex<Scalar>& Edge<Scalar,Dim>::vertex(unsigned int vert_idx) const
{
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return vertices_[vert_idx];
}

template <typename Scalar, int Dim>
Vertex<Scalar>& Edge<Scalar,Dim>::vertex(unsigned int vert_idx)
{
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return vertices_[vert_idx];
}

template <typename Scalar, int Dim>
const Vertex<Scalar>* Edge<Scalar,Dim>::vertexPtr(unsigned int vert_idx) const
{
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return &(vertices_[vert_idx]);
}

template <typename Scalar, int Dim>
Vertex<Scalar>* Edge<Scalar,Dim>::vertexPtr(unsigned int vert_idx)
{
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return &(vertices_[vert_idx]);
}

template <typename Scalar, int Dim>
void Edge<Scalar,Dim>::setEdgeNormal(const Vector<Scalar,Dim> &edge_normal)
{
    normal_ = edge_normal;
    has_normal_ = true;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> Edge<Scalar,Dim>::edgeNormal() const
{
    PHYSIKA_ASSERT(has_normal_);
    return normal_;
}

template <typename Scalar, int Dim>
bool Edge<Scalar,Dim>::hasEdgeNormal() const
{
    return has_normal_;
}

template <typename Scalar, int Dim>
void Edge<Scalar,Dim>::addVertex(const Vertex<Scalar> &vertex)
{
    vertices_.push_back(vertex);
}

template <typename Scalar, int Dim>
void Edge<Scalar,Dim>::reverseVertices()
{
    reverse(vertices_.begin(),vertices_.end());
}

template <typename Scalar, int Dim>
void Edge<Scalar,Dim>::printVertices() const
{
    for(unsigned int i = 0; i < vertices_.size(); ++i)
        std::cout<<vertices_[i].positionIndex()<<" ";
    std::cout<<"\n";
}

//explicit instantitation
template class Edge<float,2>;
template class Edge<double,2>;
template class Edge<float,3>;
template class Edge<double,3>;

} // end of namespace BoundaryMeshInternal

} // end of namespace Physika
