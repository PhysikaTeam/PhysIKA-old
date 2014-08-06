/*
 * @file group.h 
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

#ifndef PHYSIKA_GEOMETRY_POLYGON_GROUP_H_
#define PHYSIKA_GEOMETRY_POLYGON_GROUP_H_

#include <string>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Geometry/Surface_Mesh/material.h"
#include "Physika_Geometry/Polygon/edge.h"

namespace Physika{

namespace PolygonInternal{

/*
 * we assume groups can be identified by their names
 */

template <typename Scalar>
class Group
{
public:
    Group();
    ~Group();
    explicit Group(const std::string &name);
    Group(const std::string &name, const std::vector<Edge<Scalar> > &edges);
    Group(const std::string &name, unsigned int material_index);
    Group(const std::string &name, unsigned int material_index, const std::vector<Edge<Scalar> > &edges);

    unsigned int        numEdges() const;
    const std::string&  name() const;
    void                setName(const std::string &name);
    const Edge<Scalar>& edge(unsigned int edge_idx) const; 
    Edge<Scalar>&       edge(unsigned int edge_idx);
    const Edge<Scalar>* edgePtr(unsigned int edge_idx) const;
    Edge<Scalar>*       edgePtr(unsigned int edge_idx);
    unsigned int        materialIndex() const;
    void                setMaterialIndex(unsigned int material_index);
    void                addEdge(const Edge<Scalar> &edge);
    void                removeEdge(unsigned int edge_idx);
protected:
    std::string name_;
    unsigned int material_index_; //materil index in the mesh
    std::vector<Edge<Scalar> > edges_;
};

} //end of namespace PolygonInternal

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_POLYGON_GROUP_H_
