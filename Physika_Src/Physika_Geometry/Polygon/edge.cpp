/*
 * @file edge.cpp 
 * @brief edge of 2d polygon
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
#include "Physika_Geometry/Polygon/edge.h"
using std::vector;

namespace Physika{

namespace PolygonInternal{

template<typename Scalar>
Edge<Scalar>::Edge():has_normal_(false){}

template<typename Scalar>
Edge<Scalar>::~Edge(){}

template <typename Scalar>
Edge<Scalar>::Edge(const vector<Vertex<Scalar> > &vertices)
    :has_normal_(false)
{
    for(int i = 0; i < vertices.size(); ++i)
	addVertex(vertices[i]);
}

template <typename Scalar>
Edge<Scalar>::Edge(const vector<Vertex<Scalar> > &vertices, const Vector<Scalar,2> &edge_normal)
    :normal_(edge_normal),has_normal_(true)
{
    for(int i = 0; i < vertices.size(); ++i)
	addVertex(vertices[i]);
}

template <typename Scalar>
unsigned int Edge<Scalar>::numVertices() const
{
    return vertices_.size();
}

template <typename Scalar>
const Vertex<Scalar>& Edge<Scalar>::vertex(unsigned int vert_idx) const
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return vertices_[vert_idx];
}

template <typename Scalar>
Vertex<Scalar>& Edge<Scalar>::vertex(unsigned int vert_idx)
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return vertices_[vert_idx];
}

template <typename Scalar>
const Vertex<Scalar>* Edge<Scalar>::vertexPtr(unsigned int vert_idx) const
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return &(vertices_[vert_idx]);
}

template <typename Scalar>
Vertex<Scalar>* Edge<Scalar>::vertexPtr(unsigned int vert_idx)
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return &(vertices_[vert_idx]);
}

template <typename Scalar>
void Edge<Scalar>::setEdgeNormal(const Vector<Scalar,2> &edge_normal)
{
    normal_ = edge_normal;
    has_normal_ = true;
}

template <typename Scalar>
const Vector<Scalar,2>& Edge<Scalar>::edgeNormal() const
{
    PHYSIKA_ASSERT(has_normal_);
    return normal_;
}

template <typename Scalar>
bool Edge<Scalar>::hasEdgeNormal() const
{
    return has_normal_;
}

template <typename Scalar>
void Edge<Scalar>::addVertex(const Vertex<Scalar> &vertex)
{
    vertices_.push_back(vertex);
}

template <typename Scalar>
void Edge<Scalar>::reverseVertices()
{
    reverse(vertices_.begin(),vertices_.end());
}

template <typename Scalar>
void Edge<Scalar>::printVertices() const
{
    for(unsigned int i = 0; i < vertices_.size(); ++i)
        std::cout<<vertices_[i].positionIndex()<<" ";
    std::cout<<"\n";
}

//explicit instantitation
template class Edge<float>;
template class Edge<double>;

} // end of namespace SurfaceMeshInternal

} // end of namespace Physika