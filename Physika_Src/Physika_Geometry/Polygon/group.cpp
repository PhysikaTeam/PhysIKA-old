/*
 * @file group.cpp 
 * @brief edge group of 2d polygon
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
#include "Physika_Geometry/Polygon/group.h"
using std::string;
using std::vector;

namespace Physika{

namespace PolygonInternal{

template <typename Scalar>
Group<Scalar>::Group()
    :name_("NONAME"),material_index_(0)
{
}

template <typename Scalar>
Group<Scalar>::~Group()
{
}

template <typename Scalar>
Group<Scalar>::Group(const string &name)
    :name_(name),material_index_(0)
{
}

template <typename Scalar>
Group<Scalar>::Group(const string &name, const vector<Edge<Scalar> > &edges)
    :name_(name),material_index_(0)
{
    for(int i = 0; i < edges.size(); ++i)
        addEdge(edges[i]);
}

template <typename Scalar>
Group<Scalar>::Group(const string &name, unsigned int material_index)
    :name_(name),material_index_(material_index)
{
}

template <typename Scalar>
Group<Scalar>::Group(const string &name, unsigned int material_index, const vector<Edge<Scalar> > &edges)
    :name_(name),material_index_(material_index)
{
    for(int i = 0; i < edges.size(); ++i)
        addEdge(edges[i]);
}

template <typename Scalar>
unsigned int Group<Scalar>::numEdges() const
{
    return edges_.size();
}

template <typename Scalar>
const string& Group<Scalar>::name() const
{
    return name_;
}

template <typename Scalar>
void Group<Scalar>::setName(const string &name)
{
    name_ = name;
}

template <typename Scalar>
const Edge<Scalar>& Group<Scalar>::edge(unsigned int edge_idx) const
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return edges_[edge_idx];
}

template <typename Scalar>
Edge<Scalar>& Group<Scalar>::edge(unsigned int edge_idx)
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return edges_[edge_idx];
}

template <typename Scalar>
const Edge<Scalar>* Group<Scalar>::edgePtr(unsigned int edge_idx) const
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return &(edges_[edge_idx]);
}

template <typename Scalar>
Edge<Scalar>* Group<Scalar>::edgePtr(unsigned int edge_idx)
{
    PHYSIKA_ASSERT(edge_idx>=0);
    PHYSIKA_ASSERT(edge_idx<edges_.size());
    return &(edges_[edge_idx]);
}

template <typename Scalar>
unsigned int Group<Scalar>::materialIndex() const
{
    return material_index_;
}

template <typename Scalar>
void Group<Scalar>::setMaterialIndex(unsigned int material_index)
{
    material_index_ = material_index;
}

template <typename Scalar>
void Group<Scalar>::addEdge(const Edge<Scalar> &edge)
{
    edges_.push_back(edge);
}

template <typename Scalar>
void Group<Scalar>::removeEdge(unsigned int edge_idx)
{
    typename vector<Edge<Scalar> >::iterator iter = edges_.begin() + edge_idx;
    if(iter != edges_.end())
        edges_.erase(iter);
}

//explicit instantitation
template class Group<float>;
template class Group<double>;

} // end of namespace PolygonInternal

} // end of namespace Physika