/*
 * @file polygon.h 
 * @brief 2d polygon
 * @author Fei Zhu, Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_GEOMETRY_BOUNDARY_MESHES_POLYGON_H_
#define PHYSIKA_GEOMETRY_BOUNDARY_MESHES_POLYGON_H_

#include <vector>
#include <string>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Geometry/Boundary_Meshes/boundary_mesh.h"
#include "Physika_Geometry/Boundary_Meshes/vertex.h"
#include "Physika_Geometry/Boundary_Meshes/edge.h"
#include "Physika_Geometry/Boundary_Meshes/edge_group.h"
#include "Physika_Geometry/Boundary_Meshes/material.h"

namespace Physika{

using BoundaryMeshInternal::Vertex;
using BoundaryMeshInternal::Edge;
using BoundaryMeshInternal::EdgeGroup;
using BoundaryMeshInternal::Material;

template <typename Scalar>
class Polygon: public BoundaryMesh
{
public:
    //constructors && deconstructors
    Polygon();
    ~Polygon();

    //basic info
    unsigned int dims() const {return 2;}
    unsigned int numVertices() const;
    unsigned int numEdges() const;
    unsigned int numNormals() const;
    unsigned int numTextureCoordinates() const;
    unsigned int numGroups() const;
    unsigned int numIsolatedVertices() const;
    bool isTriangularPolygon() const;
    bool isQuadrilateralPolygon() const;
    
    //getters && setters
    const Vector<Scalar,2>& vertexPosition(unsigned int vert_idx) const;
    const Vector<Scalar,2>& vertexPosition(const Vertex<Scalar> &vertex) const;
    void                    setVertexPosition(unsigned int vert_idx, const Vector<Scalar,2> &position);
    void                    setVertexPosition(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &position);
    const Vector<Scalar,2>& vertexNormal(unsigned int normal_idx) const;
    const Vector<Scalar,2>& vertexNormal(const Vertex<Scalar> &vertex) const;
    void                    setVertexNormal(unsigned int normal_idx, const Vector<Scalar,2> &normal);
    void                    setVertexNormal(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &normal);
    const Vector<Scalar,2>& vertexTextureCoordinate(unsigned int texture_idx) const;
    const Vector<Scalar,2>& vertexTextureCoordinate(const Vertex<Scalar> &vertex) const;
    void                    setVertexTextureCoordinate(unsigned int texture_idx, const Vector<Scalar,2> &texture_coordinate);
    void                    setVertexTextureCoordinate(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &texture_coordinate);
    const EdgeGroup<Scalar,2>&    group(unsigned int group_idx) const;
    EdgeGroup<Scalar,2>&          group(unsigned int group_idx);
    const EdgeGroup<Scalar,2>*    groupPtr(unsigned int group_idx) const;
    EdgeGroup<Scalar,2>*          groupPtr(unsigned int group_idx);
    const EdgeGroup<Scalar,2>*    groupPtr(const std::string &name) const;
    EdgeGroup<Scalar,2>*          groupPtr(const std::string &name);
    const Material<Scalar>& material() const;
    Material<Scalar>&       material();
    const Material<Scalar>* materialPtr() const;
    Material<Scalar>*       materialPtr();
    void                    setMaterial(const Material<Scalar> &material);
    const Edge<Scalar,2>&     edge(unsigned int edge_idx) const; 
    Edge<Scalar,2>&           edge(unsigned int edge_idx); 
    const Edge<Scalar,2>*     edgePtr(unsigned int edge_idx) const; 
    Edge<Scalar,2>*           edgePtr(unsigned int edge_idx); 

    //adders
    void addGroup(const EdgeGroup<Scalar,2> &group);
    void addVertexPosition(const Vector<Scalar,2> &position);
    void addVertexNormal(const Vector<Scalar,2> &normal);
    void addVertexTextureCoordinate(const Vector<Scalar,2> &texture_coordinate);

    //utilities
    enum VertexNormalType{
    WEIGHTED_EDGE_NORMAL = 0,
    AVERAGE_EDGE_NORMAL = 1,
    EDGE_NORMAL = 2};

    void computeAllVertexNormals(VertexNormalType normal_type);
    void computeAllEdgeNormals();
    void computeEdgeNormal(Edge<Scalar,2> &edge);

protected:
    void setVertexNormalsToEdgeNormals();
    void setVertexNormalsToAverageEdgeNormals();
    void setVertexNormalsToWeightedEdgeNormals();//weight edge normal with angle

protected:
    std::vector<Vector<Scalar,2> > vertex_positions_;
    std::vector<Vector<Scalar,2> > vertex_normals_;
    std::vector<Vector<Scalar,2> > vertex_textures_;
    std::vector<EdgeGroup<Scalar,2> > groups_;
    Material<Scalar> material_;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_BOUNDARY_MESHES_POLYGON_H_
