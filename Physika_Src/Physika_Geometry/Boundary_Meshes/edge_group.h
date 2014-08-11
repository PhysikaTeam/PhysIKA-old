/*
 * @file group.h 
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

#ifndef PHYSIKA_GEOMETRY_BOUNDARY_MESHES_EDGE_GROUP_H_
#define PHYSIKA_GEOMETRY_BOUNDARY_MESHES_EDGE_GROUP_H_

#include <string>
#include <vector>
#include "Physika_Geometry/Boundary_Meshes/edge.h"

namespace Physika{

namespace BoundaryMeshInternal{

/*
 * we assume groups can be identified by their names
 */

template <typename Scalar, int Dim>
class EdgeGroup
{
public:
    EdgeGroup();
    ~EdgeGroup();
    explicit EdgeGroup(const std::string &name);
    EdgeGroup(const std::string &name, const std::vector<Edge<Scalar,Dim> > &edges);

    unsigned int        numEdges() const;
    const std::string&  name() const;
    void                setName(const std::string &name);
    const Edge<Scalar,Dim>& edge(unsigned int edge_idx) const; 
    Edge<Scalar,Dim>&       edge(unsigned int edge_idx);
    const Edge<Scalar,Dim>* edgePtr(unsigned int edge_idx) const;
    Edge<Scalar,Dim>*       edgePtr(unsigned int edge_idx);
    void                addEdge(const Edge<Scalar,Dim> &edge);
    void                removeEdge(unsigned int edge_idx);
protected:
    std::string name_;
    std::vector<Edge<Scalar,Dim> > edges_;
};

} //end of namespace BoundaryMeshInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BOUNDARY_MESHES_GROUP_H_
