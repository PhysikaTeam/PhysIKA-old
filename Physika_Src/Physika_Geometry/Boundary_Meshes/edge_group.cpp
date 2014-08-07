/*
 * @file group.cpp 
 * @brief edge group of 2d polygon and 3d surface mesh
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

#include <cstddef> 
#include <algorithm>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Boundary_Meshes/edge_group.h"
using std::string;
using std::vector;

namespace Physika{

namespace BoundaryMeshInternal{

template <typename Scalar, int Dim>
EdgeGroup<Scalar,Dim>::EdgeGroup()
    :name_("NONAME")
{
}

template <typename Scalar, int Dim>
EdgeGroup<Scalar,Dim>::~EdgeGroup()
{
}

template <typename Scalar, int Dim>
EdgeGroup<Scalar,Dim>::EdgeGroup(const string &name)
    :name_(name)
{
}

template <typename Scalar, int Dim>
EdgeGroup<Scalar,Dim>::EdgeGroup(const string &name, const vector<Edge<Scalar,Dim> > &edges)
    :name_(name)
{
    for(int i = 0; i < edges.size(); ++i)
        addEdge(edges[i]);
}

template <typename Scalar, int Dim>
unsigned int EdgeGroup<Scalar,Dim>::numEdges() const
{
    return edges_.size();
}

template <typename Scalar, int Dim>
const string& EdgeGroup<Scalar,Dim>::name() const
{
    return name_;
}

template <typename Scalar, int Dim>
void EdgeGroup<Scalar,Dim>::setName(const string &name)
{
    name_ = name;
}

template <typename Scalar, int Dim>
const Edge<Scalar,Dim>& EdgeGroup<Scalar,Dim>::edge(unsigned int edge_idx) const
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return edges_[edge_idx];
}

template <typename Scalar, int Dim>
Edge<Scalar,Dim>& EdgeGroup<Scalar,Dim>::edge(unsigned int edge_idx)
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return edges_[edge_idx];
}

template <typename Scalar, int Dim>
const Edge<Scalar,Dim>* EdgeGroup<Scalar,Dim>::edgePtr(unsigned int edge_idx) const
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return &(edges_[edge_idx]);
}

template <typename Scalar, int Dim>
Edge<Scalar,Dim>* EdgeGroup<Scalar,Dim>::edgePtr(unsigned int edge_idx)
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return &(edges_[edge_idx]);
}

template <typename Scalar, int Dim>
void EdgeGroup<Scalar,Dim>::addEdge(const Edge<Scalar,Dim> &edge)
{
    edges_.push_back(edge);
}

template <typename Scalar, int Dim>
void EdgeGroup<Scalar,Dim>::removeEdge(unsigned int edge_idx)
{
    typename vector<Edge<Scalar,Dim> >::iterator iter = edges_.begin() + edge_idx;
    if(iter != edges_.end())
        edges_.erase(iter);
}

//explicit instantitation
template class EdgeGroup<float,2>;
template class EdgeGroup<double,2>;
template class EdgeGroup<float,3>;
template class EdgeGroup<double,3>;

} // end of namespace BoundaryMeshInternal

} // end of namespace Physika
