/*
 * @file polygon.h 
 * @brief 2d polygon
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

#ifndef PHYSIKA_GEOMETRY_POLYGON_POLYGON_H_
#define PHYSIKA_GEOMETRY_POLYGON_POLYGON_H_

#include <vector>
#include <string>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Geometry/Surface_Mesh/vertex.h"
#include "Physika_Geometry/Polygon/edge.h"
#include "Physika_Geometry/Polygon/group.h"
#include "Physika_Geometry/Surface_Mesh/material.h"

namespace Physika{

using SurfaceMeshInternal::Vertex;
using PolygonInternal::Edge;
using PolygonInternal::Group;
using SurfaceMeshInternal::Material;

template <typename Scalar>
class Polygon
{
public:
    //constructors && deconstructors
    Polygon();
    ~Polygon();

    //basic info
    unsigned int numVertices() const;
    unsigned int numEdges() const;
    unsigned int numNormals() const;
    unsigned int numTextureCoordinates() const;
    unsigned int numGroups() const;
    unsigned int numMaterials() const;
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
    const Group<Scalar>&    group(unsigned int group_idx) const;
    Group<Scalar>&          group(unsigned int group_idx);
    const Group<Scalar>*    groupPtr(unsigned int group_idx) const;
    Group<Scalar>*          groupPtr(unsigned int group_idx);
    const Group<Scalar>*    groupPtr(const std::string &name) const;
    Group<Scalar>*          groupPtr(const std::string &name);
    const Material<Scalar>& material(unsigned int material_idx) const;
    Material<Scalar>&       material(unsigned int material_idx);
    const Material<Scalar>* materialPtr(unsigned int material_idx) const;
    Material<Scalar>*       materialPtr(unsigned int material_idx);
    unsigned int            materialIndex(const std::string &material_name) const; //if no material with given name, return -1
    void                    setSingleMaterial(const Material<Scalar> &material); //set single material for entire mesh
	const Edge<Scalar>&     edge(unsigned int edge_idx) const; 
	Edge<Scalar>&           edge(unsigned int edge_idx); 
	const Edge<Scalar>*     edgePtr(unsigned int edge_idx) const; 
	Edge<Scalar>*           edgePtr(unsigned int edge_idx); 

    //adders
    void addMaterial(const Material<Scalar> &material);
    void addGroup(const Group<Scalar> &group);
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
    void computeEdgeNormal(Edge<Scalar> &edge);

protected:
    void setVertexNormalsToEdgeNormals();
    void setVertexNormalsToAverageEdgeNormals();
    void setVertexNormalsToWeightedEdgeNormals();//weight edge normal with angle

protected:
    std::vector<Vector<Scalar,2> > vertex_positions_;
    std::vector<Vector<Scalar,2> > vertex_normals_;
    std::vector<Vector<Scalar,2> > vertex_textures_;
    std::vector<Group<Scalar> > groups_;
    std::vector<Material<Scalar> > materials_;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_POLYGON_POLYGON_H_
