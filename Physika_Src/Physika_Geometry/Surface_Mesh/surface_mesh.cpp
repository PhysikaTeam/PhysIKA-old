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

#include <cmath>
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
unsigned int SurfaceMesh<Scalar>::numIsolatedVertices() const
{
    vector<unsigned int> neighor_face_count(numVertices(),0);
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
	const Group<Scalar> &group = groups_[group_idx];
	for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
	{
	    const Face<Scalar> &face = group.face(face_idx);
	    for(unsigned int vert_idx = 0; vert_idx < face.numVertices(); ++vert_idx)
	    {
		const Vertex<Scalar> &vertex = face.vertex(vert_idx);
		unsigned int pos_idx = vertex.positionIndex();
		++neighor_face_count[pos_idx];
	    }
	}
    }
    unsigned int num_isolated_vertices = 0;
    for(unsigned int i = 0; i < numVertices(); ++i)
	if(neighor_face_count[i]>0)
	    ++num_isolated_vertices;
    return num_isolated_vertices;
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
Group<Scalar>* SurfaceMesh<Scalar>::groupPtr(const string &name)
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	if(groups_[group_idx].name() == name)
	    return &(groups_[group_idx]);
    return NULL;
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
unsigned int SurfaceMesh<Scalar>::materialIndex(const string &material_name) const
{
    for(unsigned int i = 0; i < materials_.size(); ++i)
	if(materials_[i].name() == material_name)
	    return i;
    return -1;
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
    switch(normal_type)
    {
    case WEIGHTED_FACE_NORMAL:
	setVertexNormalsToWeightedFaceNormals();
	break;
    case AVERAGE_FACE_NORMAL:
	setVertexNormalsToAverageFaceNormals();
	break;
    case FACE_NORMAL:
	setVertexNormalsToFaceNormals();
	break;
    default:
	PHYSIKA_ERROR("Wrong normal type specified!");
    }
}

template <typename Scalar>
void SurfaceMesh<Scalar>::computeAllFaceNormals()
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
	Group<Scalar> &group = groups_[group_idx];
	for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
	{
	    Face<Scalar> &face = group.face(face_idx);
	    computeFaceNormal(face);
	}
    }
}

template <typename Scalar>
void SurfaceMesh<Scalar>::computeFaceNormal(Face<Scalar> &face)
{
    //cmopute face normal with the first 3 vertices
    PHYSIKA_ASSERT(face.numVertices()>=3);
    unsigned int vert_idx1 = face.vertex(0).positionIndex();
    unsigned int vert_idx2 = face.vertex(1).positionIndex();
    unsigned int vert_idx3 =  face.vertex(2).positionIndex();
    const Vector<Scalar,3> &vert_pos1 = vertexPosition(vert_idx1);
    const Vector<Scalar,3> &vert_pos2 = vertexPosition(vert_idx2);
    const Vector<Scalar,3> &vert_pos3 = vertexPosition(vert_idx3);
    Vector<Scalar,3> vec1 = vert_pos2 - vert_pos1;
    Vector<Scalar,3> vec2 = vert_pos3 - vert_pos1;
    Vector<Scalar,3> normal = (vec1.cross(vec2)).normalize();
    face.setFaceNormal(normal);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexNormalsToFaceNormals()
{
    vertex_normals_.clear();
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
	Group<Scalar> &group = groups_[group_idx];
	for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
	{
	    Face<Scalar> &face = group.face(face_idx);
	    if(!face.hasFaceNormal())
	        computeFaceNormal(face);
	    addVertexNormal(face.faceNormal());
	    for(unsigned int vert_idx = 0; vert_idx < face.numVertices(); ++vert_idx)
	    {
		Vertex<Scalar> &vertex = face.vertex(vert_idx);
		vertex.setNormalIndex(vertex_normals_.size()-1);
	    }
	}
    }
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexNormalsToAverageFaceNormals()
{
    vector<Vector<Scalar,3> > normal_buffer(numVertices(),Vector<Scalar,3>(0.0));
    vector<unsigned int> normal_count(numVertices(),0);

    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
	Group<Scalar> &group = groups_[group_idx];
	for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
	{
	    Face<Scalar> &face = group.face(face_idx);
	    if(!face.hasFaceNormal())
	        computeFaceNormal(face);
	    const Vector<Scalar,3> &face_normal = face.faceNormal();
	    for(unsigned int vert_idx = 0; vert_idx < face.numVertices(); ++vert_idx)
	    {
		Vertex<Scalar> &vertex = face.vertex(vert_idx);
		unsigned int pos_idx = vertex.positionIndex();
		normal_buffer[pos_idx] += face_normal;
		++normal_count[pos_idx];
	    }
	}
    }
    //normalize the normals and apply the new normals
    vertex_normals_.clear();
    for(unsigned int i = 0; i < numVertices(); ++i)
    {
	if(normal_count[i] == 0)//isolated vertex, use some fake normal
	    normal_buffer[i] = Vector<Scalar,3>(1.0,0,0);
	normal_buffer[i].normalize();
	addVertexNormal(normal_buffer[i]);
    }
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
	Group<Scalar> &group = groups_[group_idx];
	for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
	{
	    Face<Scalar> &face = group.face(face_idx);
	    for(unsigned int vert_idx = 0; vert_idx < face.numVertices(); ++vert_idx)
	    {
		Vertex<Scalar> &vertex = face.vertex(vert_idx);
		vertex.setNormalIndex(vertex.positionIndex());
	    }
	}
    }
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexNormalsToWeightedFaceNormals()
{
    vector<Vector<Scalar,3> > normal_buffer(numVertices(),Vector<Scalar,3>(0.0));
    vector<unsigned int> normal_count(numVertices(),0);

    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
	Group<Scalar> &group = groups_[group_idx];
	for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
	{
	    Face<Scalar> &face = group.face(face_idx);
	    if(!face.hasFaceNormal())
	        computeFaceNormal(face);
	    const Vector<Scalar,3> &face_normal = face.faceNormal();
	    for(unsigned int vert_idx = 0; vert_idx < face.numVertices(); ++vert_idx)
	    {
		Vertex<Scalar> &vertex = face.vertex(vert_idx);
		Vertex<Scalar> &next_vertex = face.vertex((vert_idx+1)%face.numVertices());
		Vertex<Scalar> &pre_vertex = face.vertex((vert_idx-1+face.numVertices())%face.numVertices());
		Vector<Scalar,3> vec1 = vertex_positions_[next_vertex.positionIndex()]-vertex_positions_[vertex.positionIndex()];
		Vector<Scalar,3> vec2 = vertex_positions_[pre_vertex.positionIndex()]-vertex_positions_[vertex.positionIndex()];
		Scalar vec1_length = vec1.norm(), vec2_length = vec2.norm();
		PHYSIKA_ASSERT(vec1_length>0);
		PHYSIKA_ASSERT(vec2_length>0);
		Scalar angle = acos(vec1.dot(vec2)/(vec1_length*vec2_length));
		unsigned int pos_idx = vertex.positionIndex();
		normal_buffer[pos_idx] += angle*face_normal;
		++normal_count[pos_idx];
	    }
	}
    }
    //normalize the normals and apply the new normals
    vertex_normals_.clear();
    for(unsigned int i = 0; i < numVertices(); ++i)
    {
	if(normal_count[i] == 0)//isolated vertex, use some fake normal
	    normal_buffer[i] = Vector<Scalar,3>(1.0,0,0);
	normal_buffer[i].normalize();
	addVertexNormal(normal_buffer[i]);
    }
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
	Group<Scalar> &group = groups_[group_idx];
	for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
	{
	    Face<Scalar> &face = group.face(face_idx);
	    for(unsigned int vert_idx = 0; vert_idx < face.numVertices(); ++vert_idx)
	    {
		Vertex<Scalar> &vertex = face.vertex(vert_idx);
		vertex.setNormalIndex(vertex.positionIndex());
	    }
	}
    }
}

//explicit instantitation
template class SurfaceMesh<float>;
template class SurfaceMesh<double>;

} //end of namespace Physika


















