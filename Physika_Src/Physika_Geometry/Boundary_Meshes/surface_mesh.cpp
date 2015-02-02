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
#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
using std::vector;
using std::string;

namespace Physika{

using BoundaryMeshInternal::Vertex;
using SurfaceMeshInternal::Face;
using SurfaceMeshInternal::FaceGroup;
using BoundaryMeshInternal::Material;

template <typename Scalar>
SurfaceMesh<Scalar>::SurfaceMesh()
{
    Material<Scalar> material_example;//default material. we will add more material if need
    material_example.setAlpha(1);
    material_example.setKa(Vector<Scalar,3>(0.2,0.2,0.2));
    material_example.setKd(Vector<Scalar,3>(1,1,1));
    material_example.setKs(Vector<Scalar,3>(0.2,0.2,0.2));
    material_example.setName(string("default"));
    material_example.setShininess(24.2515);
    addMaterial(material_example);
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
    unsigned int num_face = 0;
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
        const FaceGroup<Scalar> &group = groups_[group_idx];
        num_face += group.numFaces();
    }
    return num_face;
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
unsigned int SurfaceMesh<Scalar>::numTextures() const
{
    unsigned int num_textures = 0;
    for(unsigned int material_idx = 0; material_idx < materials_.size(); ++material_idx)
        if(materials_[material_idx].hasTexture())
            ++num_textures;
    return num_textures;
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
        const FaceGroup<Scalar> &group = groups_[group_idx];
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
Vector<Scalar,3> SurfaceMesh<Scalar>::vertexPosition(unsigned int vert_idx) const
{
    bool index_valid = (vert_idx>=0)&&(vert_idx<vertex_positions_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_positions_[vert_idx];
}

template <typename Scalar>
Vector<Scalar,3> SurfaceMesh<Scalar>::vertexPosition(const Vertex<Scalar> &vertex) const
{
    unsigned int vert_idx = vertex.positionIndex();
    return vertexPosition(vert_idx);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexPosition(unsigned int vert_idx, const Vector<Scalar,3> &position)
{
    bool index_valid = (vert_idx>=0)&&(vert_idx<vertex_positions_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_positions_[vert_idx] = position;
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexPosition(const Vertex<Scalar> &vertex, const Vector<Scalar,3> &position)
{
    unsigned int vert_idx = vertex.positionIndex();
    setVertexPosition(vert_idx,position);
}

template <typename Scalar>
Vector<Scalar,3> SurfaceMesh<Scalar>::vertexNormal(unsigned int normal_idx) const
{
    bool index_valid = (normal_idx>=0)&&(normal_idx<vertex_normals_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh vertex normal index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_normals_[normal_idx];
}

template <typename Scalar>
Vector<Scalar,3> SurfaceMesh<Scalar>::vertexNormal(const Vertex<Scalar> &vertex) const
{
    unsigned int normal_idx = vertex.normalIndex();
    return vertexNormal(normal_idx);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexNormal(unsigned int normal_idx, const Vector<Scalar,3> &normal)
{
    bool index_valid = (normal_idx>=0)&&(normal_idx<vertex_normals_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh vertex normal index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_normals_[normal_idx] = normal;
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexNormal(const Vertex<Scalar> &vertex, const Vector<Scalar,3> &normal)
{
    unsigned int normal_idx = vertex.normalIndex();
    setVertexNormal(normal_idx,normal);
}

template <typename Scalar>
Vector<Scalar,2> SurfaceMesh<Scalar>::vertexTextureCoordinate(unsigned int texture_idx) const
{
    bool index_valid = (texture_idx>=0)&&(texture_idx<vertex_textures_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh vertex texture index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_textures_[texture_idx];
}

template <typename Scalar>
Vector<Scalar,2> SurfaceMesh<Scalar>::vertexTextureCoordinate(const Vertex<Scalar> &vertex) const
{
    unsigned int texture_idx = vertex.textureCoordinateIndex();
    return vertexTextureCoordinate(texture_idx);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexTextureCoordinate(unsigned int texture_idx, const Vector<Scalar,2> &texture_coordinate)
{
    bool index_valid = (texture_idx>=0)&&(texture_idx<vertex_textures_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh vertex texture index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_textures_[texture_idx] = texture_coordinate;
}

template <typename Scalar>
void SurfaceMesh<Scalar>::setVertexTextureCoordinate(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &texture_coordinate)
{
    unsigned int texture_idx = vertex.textureCoordinateIndex();
    setVertexTextureCoordinate(texture_idx,texture_coordinate);
}

template <typename Scalar>
const FaceGroup<Scalar>& SurfaceMesh<Scalar>::group(unsigned int group_idx) const
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return groups_[group_idx];
}

template <typename Scalar>
FaceGroup<Scalar>& SurfaceMesh<Scalar>::group(unsigned int group_idx)
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return groups_[group_idx];
}

template <typename Scalar>
const FaceGroup<Scalar>* SurfaceMesh<Scalar>::groupPtr(unsigned int group_idx) const
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return &(groups_[group_idx]);
}

template <typename Scalar>
FaceGroup<Scalar>* SurfaceMesh<Scalar>::groupPtr(unsigned int group_idx)
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return &(groups_[group_idx]);
}

template <typename Scalar>
const FaceGroup<Scalar>* SurfaceMesh<Scalar>::groupPtr(const string &name) const
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
        if(groups_[group_idx].name() == name)
            return &(groups_[group_idx]);
    return NULL;
}

template <typename Scalar>
FaceGroup<Scalar>* SurfaceMesh<Scalar>::groupPtr(const string &name)
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
        if(groups_[group_idx].name() == name)
            return &(groups_[group_idx]);
    return NULL;
}

template <typename Scalar>
const Material<Scalar>& SurfaceMesh<Scalar>::material(unsigned int material_idx) const
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return materials_[material_idx];
}

template <typename Scalar>
Material<Scalar>& SurfaceMesh<Scalar>::material(unsigned int material_idx)
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return materials_[material_idx];
}

template <typename Scalar>
const Material<Scalar>* SurfaceMesh<Scalar>::materialPtr(unsigned int material_idx) const
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return &(materials_[material_idx]);
}

template <typename Scalar>
Material<Scalar>* SurfaceMesh<Scalar>::materialPtr(unsigned int material_idx)
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
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
const Face<Scalar>& SurfaceMesh<Scalar>::face(unsigned int face_idx) const 
{
    bool index_valid = (face_idx>=0)&&(face_idx<numFaces());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh face index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_face_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		const FaceGroup<Scalar>& current_group = groups_[group_idx];
		unsigned int group_face_num = current_group.numFaces();
		if(current_face_sum + group_face_num > face_idx)//find the group containing this face
		{
			return current_group.face(face_idx - current_face_sum);
		}
		else
			current_face_sum += group_face_num;
	}
	std::cerr<<"SurfaceMesh face index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].face(0);
}

template <typename Scalar>
Face<Scalar>& SurfaceMesh<Scalar>::face(unsigned int face_idx) 
{
    bool index_valid = (face_idx>=0)&&(face_idx<numFaces());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh face index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_face_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		FaceGroup<Scalar>& current_group = groups_[group_idx];
		unsigned int group_face_num = current_group.numFaces();
		if(current_face_sum + group_face_num > face_idx)//find the group containing this face
		{
			return current_group.face(face_idx - current_face_sum);
		}
		else
			current_face_sum += group_face_num;
	}
	std::cerr<<"SurfaceMesh face index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].face(0);
}

template <typename Scalar>
const Face<Scalar>* SurfaceMesh<Scalar>::facePtr(unsigned int face_idx) const 
{
    bool index_valid = (face_idx>=0)&&(face_idx<numFaces());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh face index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_face_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		const FaceGroup<Scalar>& current_group = groups_[group_idx];
		unsigned int group_face_num = current_group.numFaces();
		if(current_face_sum + group_face_num > face_idx)//find the group containing this face
		{
			return current_group.facePtr(face_idx - current_face_sum);
		}
		else
			current_face_sum += group_face_num;
	}
	std::cerr<<"SurfaceMesh face index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].facePtr(0);
}

template <typename Scalar>
Face<Scalar>* SurfaceMesh<Scalar>::facePtr(unsigned int face_idx) 
{
    bool index_valid = (face_idx>=0)&&(face_idx<numFaces());
    if(!index_valid)
    {
        std::cerr<<"SurfaceMesh face index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_face_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		FaceGroup<Scalar>& current_group = groups_[group_idx];
		unsigned int group_face_num = current_group.numFaces();
		if(current_face_sum + group_face_num > face_idx)//find the group containing this face
		{
			return current_group.facePtr(face_idx - current_face_sum);
		}
		else
			current_face_sum += group_face_num;
	}
	std::cerr<<"SurfaceMesh face index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].facePtr(0);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::addMaterial(const Material<Scalar> &material)
{
    materials_.push_back(material);
}

template <typename Scalar>
void SurfaceMesh<Scalar>::addGroup(const FaceGroup<Scalar> &group)
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
        std::cerr<<"Wrong normal type specified, use weighted face nromals!\n";
        setVertexNormalsToWeightedFaceNormals();
    }
}

template <typename Scalar>
void SurfaceMesh<Scalar>::computeAllFaceNormals()
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
        FaceGroup<Scalar> &group = groups_[group_idx];
        for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
        {
            Face<Scalar> &face = group.face(face_idx);
            computeFaceNormal(face);
        }
    }
}

template <typename Scalar>
void SurfaceMesh<Scalar>::separateByGroup(std::vector<SurfaceMesh<Scalar> > & surface_mesh_vec) const
{
	surface_mesh_vec.clear();
	for(unsigned int group_id = 0; group_id < this->numGroups(); group_id ++)
		surface_mesh_vec.push_back(SurfaceMesh());

	// the following variable(flag) is used to save the new index of old index in new mesh
	long * vertex_new_pos = new long[this->numVertices()];
	long * vertex_new_normals = new long[this->numNormals()];
	long * vertex_new_textures = new long[this->numTextureCoordinates()];

	// initial flag to -1, which means that old index having no new index in new mesh
	for(unsigned int group_id = 0; group_id < this->numGroups(); group_id++)
	{
		for(unsigned int i=0; i<this->numVertices(); i++)
			vertex_new_pos[i] = -1;
		for(unsigned int i=0; i<this->numNormals(); i++)
			vertex_new_normals[i] = -1;
		for(unsigned int i=0; i<this->numTextureCoordinates(); i++)
			vertex_new_textures[i] = -1;

		SurfaceMesh<Scalar> &     new_mesh_ref = surface_mesh_vec[group_id];  // get new mesh reference
		new_mesh_ref.addGroup(FaceGroup<Scalar>());                           // add group to new mesh, each new mesh exists only one group
		
		FaceGroup<Scalar> &       new_group_ref = new_mesh_ref.group(0);      // get new group reference 
		const FaceGroup<Scalar> & group_ref = this->group(group_id);          // get old group reference

		new_mesh_ref.addMaterial(this->material(group_ref.materialIndex()));  // add material
		new_group_ref.setName(group_ref.name());                              // set group name
		new_group_ref.setMaterialIndex(1);                                    // set group material index

		unsigned int face_num = group_ref.numFaces();
		for(unsigned int face_id = 0; face_id<face_num; face_id++)
		{
			new_group_ref.addFace(Face<Scalar>());      // add new face
			Face<Scalar> & new_face_ref = new_group_ref.face(new_group_ref.numFaces()-1);  // get last face reference of new group

			const Face<Scalar> face_ref = group_ref.face(face_id);                         // get old face reference
			if(face_ref.hasFaceNormal())               // set face normal
			{
				new_face_ref.setFaceNormal(face_ref.faceNormal());
			}

			unsigned int vertex_num = face_ref.numVertices();
			for(unsigned int vertex_id = 0; vertex_id<vertex_num; vertex_id++)
			{
				new_face_ref.addVertex(Vertex<Scalar>()); // add new vertex

				Vertex<Scalar> & new_vertex_ref = new_face_ref.vertex(new_face_ref.numVertices()-1); // get new vertex reference
				const Vertex<Scalar> & vertex_ref = face_ref.vertex(vertex_id);                      // get old vertex reference

				unsigned int pos_index = vertex_ref.positionIndex(); // save position and corresponding index
				if(vertex_new_pos[pos_index] == -1)
				{
					new_mesh_ref.addVertexPosition(this->vertexPosition(pos_index));
					vertex_new_pos[pos_index] = new_mesh_ref.numVertices()-1;
				}
				new_vertex_ref.setPositionIndex(vertex_new_pos[pos_index]);
				

				if(vertex_ref.hasNormal()) // if has vertex normal
				{
					unsigned int normal_index = vertex_ref.normalIndex();
					if(vertex_new_normals[normal_index] == -1)
					{
						new_mesh_ref.addVertexNormal(this->vertexNormal(normal_index));
						vertex_new_normals[normal_index] = new_mesh_ref.numNormals()-1;
					}
					new_vertex_ref.setNormalIndex(vertex_new_normals[normal_index]);
				}

				if(vertex_ref.hasTexture()) // if has texture
				{
					unsigned int texture_index = vertex_ref.textureCoordinateIndex();
					if(vertex_new_textures[texture_index] == -1)
					{
						new_mesh_ref.addVertexTextureCoordinate(this->vertexTextureCoordinate(texture_index));
						vertex_new_textures[texture_index] = new_mesh_ref.numTextureCoordinates()-1;
					}
					new_vertex_ref.setTextureCoordinateIndex(vertex_new_textures[texture_index]);
				}
			}
		}
	}

	// free flag
	delete [] vertex_new_pos;
	delete [] vertex_new_normals;
	delete [] vertex_new_textures;

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
        FaceGroup<Scalar> &group = groups_[group_idx];
        for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
        {
            Face<Scalar> &face = group.face(face_idx);
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
        FaceGroup<Scalar> &group = groups_[group_idx];
        for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
        {
            Face<Scalar> &face = group.face(face_idx);
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
        FaceGroup<Scalar> &group = groups_[group_idx];
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
        FaceGroup<Scalar> &group = groups_[group_idx];
        for(unsigned int face_idx = 0; face_idx < group.numFaces(); ++face_idx)
        {
            Face<Scalar> &face = group.face(face_idx);
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
        FaceGroup<Scalar> &group = groups_[group_idx];
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
