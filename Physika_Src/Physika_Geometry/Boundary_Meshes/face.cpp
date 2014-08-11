/*
 * @file face.cpp 
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

#include <iostream>
#include <algorithm>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Boundary_Meshes/face.h"
using std::vector;

namespace Physika{

namespace SurfaceMeshInternal{

template <typename Scalar>
Face<Scalar>::Face()
    :has_normal_(false)
{
}

template <typename Scalar>
Face<Scalar>::~Face()
{
}

template <typename Scalar>
Face<Scalar>::Face(const vector<Vertex<Scalar> > &vertices)
    :has_normal_(false)
{
    for(int i = 0; i < vertices.size(); ++i)
        addVertex(vertices[i]);
}

template <typename Scalar>
Face<Scalar>::Face(const vector<Vertex<Scalar> > &vertices, const Vector<Scalar,3> &face_normal)
    :normal_(face_normal),has_normal_(true)
{
    for(int i = 0; i < vertices.size(); ++i)
        addVertex(vertices[i]);
}

template <typename Scalar>
unsigned int Face<Scalar>::numVertices() const
{
    return vertices_.size();
}

template <typename Scalar>
const Vertex<Scalar>& Face<Scalar>::vertex(unsigned int vert_idx) const
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return vertices_[vert_idx];
}

template <typename Scalar>
Vertex<Scalar>& Face<Scalar>::vertex(unsigned int vert_idx)
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return vertices_[vert_idx];
}

template <typename Scalar>
const Vertex<Scalar>* Face<Scalar>::vertexPtr(unsigned int vert_idx) const
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return &(vertices_[vert_idx]);
}

template <typename Scalar>
Vertex<Scalar>* Face<Scalar>::vertexPtr(unsigned int vert_idx)
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertices_.size());
    return &(vertices_[vert_idx]);
}

template <typename Scalar>
void Face<Scalar>::setFaceNormal(const Vector<Scalar,3> &face_normal)
{
    normal_ = face_normal;
    has_normal_ = true;
}

template <typename Scalar>
Vector<Scalar,3> Face<Scalar>::faceNormal() const
{
    PHYSIKA_ASSERT(has_normal_);
    return normal_;
}

template <typename Scalar>
bool Face<Scalar>::hasFaceNormal() const
{
    return has_normal_;
}

template <typename Scalar>
void Face<Scalar>::addVertex(const Vertex<Scalar> &vertex)
{
    vertices_.push_back(vertex);
}

template <typename Scalar>
void Face<Scalar>::reverseVertices()
{
    reverse(vertices_.begin(),vertices_.end());
}

template <typename Scalar>
void Face<Scalar>::printVertices() const
{
    for(unsigned int i = 0; i < vertices_.size(); ++i)
        std::cout<<vertices_[i].positionIndex()<<" ";
    std::cout<<"\n";
}

//explicit instantitation
template class Face<float>;
template class Face<double>;

} //end of namespace SurfaceMeshInternal

} //end of namespace Physika
