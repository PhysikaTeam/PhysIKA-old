/*
 * @file surface_mesh.cpp 
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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"

namespace Physika{

template <typename Scalar>
SurfaceMesh<Scalar>::SurfaceMesh()
{
//TO DO: Add default material
}

template <typename Scalar>
SurfaceMesh<Scalar>::~SurfaceMesh()
{
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::numVertices() const
{
    return vertex_positions_.size();
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::numFaces() const
{
    return 0;
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::numNormals() const
{
    return vertex_normals_.size();
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::numTextureCoordinates() const
{
    return vertex_textures_.size();
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::numGroups() const
{
    return groups_.size();
}

template <typename Scalar>
unsigned int SurfaceMesh<Scalar>::numMaterials() const
{
    return materials_.size();
}

template <typename Scalar>
bool SurfaceMesh<Scalar>::isTriangularMesh() const
{
    for(unsigned int i = 0; i < groups_.size(); ++i)
	for(unsigned int j = 0; j < groups_[i].numFaces(); ++j)
	    if(groups_[i].face(j).numVertices() != 3)
		return false;
    return true;
}

template <typename Scalar>
bool SurfaceMesh<Scalar>::isQuadrilateralMesh() const
{
    for(unsigned int i = 0; i < groups_.size(); ++i)
	for(unsigned int j = 0; j < groups_[i].numFaces(); ++j)
	    if(groups_[i].face(j).numVertices() != 4)
		return false;
    return true;
}

template <typename Scalar>
const Vector<Scalar,3>& SurfaceMesh<Scalar>::vertexPosition(unsigned int vert_idx) const
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertex_positions_.size());
    return vertex_positions_[vert_idx];
}

template <typename Scalar>
const Vector<Scalar,3>& SurfaceMesh<Scalar>::vertexPosition(const Vertex<Scalar> &vertex) const
{
    unsigned int vert_idx = vertex.positionIndex();
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertex_positions_.size());
    return vertex_positions_[vert_idx];
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexPosition(unsigned int vert_idx, const Vector<Scalar,3> &position)
{
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertex_positions_.size());
    vertex_positions_[vert_idx] = position;
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexPosition(const Vertex<Scalar> &vertex, const Vector<Scalar,3> &position)
{
    unsigned int vert_idx = vertex.positionIndex();
    PHYSIKA_ASSERT(vert_idx>=0);
    PHYSIKA_ASSERT(vert_idx<vertex_positions_.size());
    vertex_positions_[vert_idx] = position;
}

template <typename Scalar>
const Vector<Scalar,3>& SurfaceMesh<Scalar>::vertexNormal(unsigned int normal_idx) const
{
    PHYSIKA_ASSERT(normal_idx>=0);
    PHYSIKA_ASSERT(normal_idx<vertex_normals_.size());
    return vertex_normals_[normal_idx];
}

template <typename Scalar>
const Vector<Scalar,3>& SurfaceMesh<Scalar>::vertexNormal(const Vertex<Scalar> &vertex) const
{
    unsigned int normal_idx = vertex.normalIndex();
    PHYSIKA_ASSERT(normal_idx>=0);
    PHYSIKA_ASSERT(normal_idx<vertex_normals_.size());
    return vertex_normals_[normal_idx];
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexNormal(unsigned int normal_idx, const Vector<Scalar,3> &normal)
{
    PHYSIKA_ASSERT(normal_idx>=0);
    PHYSIKA_ASSERT(normal_idx<vertex_normals_.size());
    vertex_normals_[normal_idx] = normal;
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexNormal(const Vertex<Scalar> &vertex, const Vector<Scalar,3> &normal)
{
    unsigned int normal_idx = vertex.normalIndex();
    PHYSIKA_ASSERT(normal_idx>=0);
    PHYSIKA_ASSERT(normal_idx<vertex_normals_.size());
    vertex_normals_[normal_idx] = normal;
}

template <typename Scalar>
const Vector<Scalar,2>& SurfaceMesh<Scalar>::vertexTextureCoordinate(unsigned int texture_idx) const
{
    PHYSIKA_ASSERT(texture_idx>=0);
    PHYSIKA_ASSERT(texture_idx<vertex_textures_.size());
    return vertex_textures_[texture_idx];
}

template <typename Scalar>
const Vector<Scalar,2>& SurfaceMesh<Scalar>::vertexTextureCoordinate(const Vertex<Scalar> &vertex) const
{
    unsigned int texture_idx = vertex.textureCoordinateIndex();
    PHYSIKA_ASSERT(texture_idx>=0);
    PHYSIKA_ASSERT(texture_idx<vertex_textures_.size());
    return vertex_textures_[texture_idx];
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexTextureCoordinate(unsigned int texture_idx, const Vector<Scalar,2> &texture_coordinate)
{
    PHYSIKA_ASSERT(texture_idx>=0);
    PHYSIKA_ASSERT(texture_idx<vertex_textures_.size());
    vertex_textures_[texture_idx] = texture_coordinate;
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexTextureCoordinate(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &texture_coordinate)
{
    unsigned int texture_idx = vertex.textureCoordinateIndex();
    PHYSIKA_ASSERT(texture_idx>=0);
    PHYSIKA_ASSERT(texture_idx<vertex_textures_.size());
    vertex_textures_[texture_idx] = texture_coordinate;
}

template <typename Scalar>
Group<Scalar>& SurfaceMesh<Scalar>::group(unsigned int group_idx)
{
    PHYSIKA_ASSERT(group_idx>=0);
    PHYSIKA_ASSERT(group_idx<groups_.size());
    return groups_[group_idx];
}

template <typename Scalar>
Group<Scalar>* SurfaceMesh<Scalar>::groupPtr(unsigned int group_idx)
{
    PHYSIKA_ASSERT(group_idx>=0);
    PHYSIKA_ASSERT(group_idx<groups_.size());
    return &(groups_[group_idx]);
}

template <typename Scalar>
Material<Scalar>& SurfaceMesh<Scalar>::material(unsigned int material_idx)
{
    PHYSIKA_ASSERT(material_idx>=0);
    PHYSIKA_ASSERT(material_idx<materials_.size());
    return materials_[material_idx];
}

template <typename Scalar>
Material<Scalar>* SurfaceMesh<Scalar>::materialPtr(unsigned int material_idx)
{
    PHYSIKA_ASSERT(material_idx>=0);
    PHYSIKA_ASSERT(material_idx<materials_.size());
    return &(materials_[material_idx]);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setSingleMaterial(const Material<Scalar> &material)
{
    materials_.clear();
    materials_.push_back(material);
    for(int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	groups_[group_idx].setMaterialIndex(0);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::addMaterial(const Material<Scalar> &material)
{
    materials_.push_back(material);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::addGroup(const Group<Scalar> &group)
{
    groups_.push_back(group);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::addVertexPosition(const Vector<Scalar,3> &position)
{
    vertex_positions_.push_back(position);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::addVertexNormal(const Vector<Scalar,3> &normal)
{
    vertex_normals_.push_back(normal);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::addVertexTextureCoordinate(const Vector<Scalar,2> &texture_coordinate)
{
    vertex_textures_.push_back(texture_coordinate);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::computeAllVertexNormals(VertexNormalType normal_type)
{

}

template <typename Scalar>
void SurfaceMesh<Scalar>::computeAllFaceNormals()
{
}

template <typename Scalar>
void SurfaceMesh<Scalar>::computeFaceNormal(const Face<Scalar> &face)
{
}

//explicit instantitation
template class SurfaceMesh<float>;
template class SurfaceMesh<double>;

} //end of namespace Physika


















