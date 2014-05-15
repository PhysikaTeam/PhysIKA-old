/*
 * @file surface_mesh.h 
 * @brief 3d surface mesh
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

#ifndef PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_
#define PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_

#include <vector>
#include <string>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Geometry/Surface_Mesh/vertex.h"
#include "Physika_Geometry/Surface_Mesh/face.h"
#include "Physika_Geometry/Surface_Mesh/group.h"
#include "Physika_Geometry/Surface_Mesh/material.h"

namespace Physika{

using SurfaceMeshInternal::Vertex;
using SurfaceMeshInternal::Face;
using SurfaceMeshInternal::Group;
using SurfaceMeshInternal::Material;

template <typename Scalar>
class SurfaceMesh
{
public:
    //constructors && deconstructors
    SurfaceMesh();
    ~SurfaceMesh();

    //basic info
    unsigned int numVertices() const;
    unsigned int numFaces() const;
    unsigned int numNormals() const;
    unsigned int numTextureCoordinates() const;
    unsigned int numGroups() const;
    unsigned int numMaterials() const;
    unsigned int numIsolatedVertices() const;
    bool isTriangularMesh() const;
    bool isQuadrilateralMesh() const;
    
    //getters && setters
    const Vector<Scalar,3>& vertexPosition(unsigned int vert_idx) const;
    const Vector<Scalar,3>& vertexPosition(const Vertex<Scalar> &vertex) const;
    void setVertexPosition(unsigned int vert_idx, const Vector<Scalar,3> &position);
    void setVertexPosition(const Vertex<Scalar> &vertex, const Vector<Scalar,3> &position);
    const Vector<Scalar,3>& vertexNormal(unsigned int normal_idx) const;
    const Vector<Scalar,3>& vertexNormal(const Vertex<Scalar> &vertex) const;
    void setVertexNormal(unsigned int normal_idx, const Vector<Scalar,3> &normal);
    void setVertexNormal(const Vertex<Scalar> &vertex, const Vector<Scalar,3> &normal);
    const Vector<Scalar,2>& vertexTextureCoordinate(unsigned int texture_idx) const;
    const Vector<Scalar,2>& vertexTextureCoordinate(const Vertex<Scalar> &vertex) const;
    void setVertexTextureCoordinate(unsigned int texture_idx, const Vector<Scalar,2> &texture_coordinate);
    void setVertexTextureCoordinate(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &texture_coordinate);
    Group<Scalar>& group(unsigned int group_idx);
    Group<Scalar>* groupPtr(unsigned int group_idx);
    Group<Scalar>* groupPtr(const std::string &name);
    Material<Scalar>& material(unsigned int material_idx);
    Material<Scalar>* materialPtr(unsigned int material_idx);
    unsigned int materialIndex(const std::string &material_name) const; //if no material with given name, return -1
    void setSingleMaterial(const Material<Scalar> &material); //set single material for entire mesh

    //adders
    void addMaterial(const Material<Scalar> &material);
    void addGroup(const Group<Scalar> &group);
    void addVertexPosition(const Vector<Scalar,3> &position);
    void addVertexNormal(const Vector<Scalar,3> &normal);
    void addVertexTextureCoordinate(const Vector<Scalar,2> &texture_coordinate);

    //utilities
    enum VertexNormalType{
	WEIGHTED_FACE_NORMAL = 0,
	AVERAGE_FACE_NORMAL = 1,
	FACE_NORMAL = 2};

    void computeAllVertexNormals(VertexNormalType normal_type);
    void computeAllFaceNormals();
    void computeFaceNormal(Face<Scalar> &face);

protected:
    void setVertexNormalsToFaceNormals();
    void setVertexNormalsToAverageFaceNormals();
    void setVertexNormalsToWeightedFaceNormals();//weight face normal with angle

protected:
    std::vector<Vector<Scalar,3> > vertex_positions_;
    std::vector<Vector<Scalar,3> > vertex_normals_;
    std::vector<Vector<Scalar,2> > vertex_textures_;
    std::vector<Group<Scalar> > groups_;
    std::vector<Material<Scalar> > materials_;
};

} //end of namespace Physika

#endif //PHYSIKA_GEOMETRY_SURFACE_MESH_SURFACE_MESH_H_















