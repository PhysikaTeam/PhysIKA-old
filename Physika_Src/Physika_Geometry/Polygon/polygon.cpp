/*
 * @file  polygon.cpp
 * @class of 2D polygon
 * @author Tianxiang Zhang, Wei Chen
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
#include "Physika_Geometry/Polygon/polygon.h"
using std::vector;
using std::string;
namespace Physika{

template<typename Scalar>
Polygon<Scalar>::Polygon()
{
	Material<Scalar> material_example;//default material. we will add more material if need
    material_example.setAlpha(1);
    material_example.setKa(Vector<Scalar,3>(0.2,0.2,0.2));
    material_example.setKd(Vector<Scalar,3>(1,1,1));
    material_example.setKs(Vector<Scalar,3>(0.2,0.2,0.2));
    material_example.setName(string("default"));
    material_example.setShininess(24.2515);
    addMaterial(material_example);
    material_example.setKa(Vector<Scalar, 3>(0.8941, 0.8392, 0.6000));
    material_example.setKd(Vector<Scalar, 3>(0.8941, 0.8392, 0.6000));
    material_example.setKs(Vector<Scalar, 3>(0.3500, 0.3500, 0.3500));
    material_example.setName(string("default_iron"));
    material_example.setShininess(32);
    addMaterial(material_example);
}

template<typename Scalar>
Polygon<Scalar>::~Polygon(){}

template <typename Scalar>
unsigned int Polygon<Scalar>::numVertices() const
{
	return this->vertex_positions_.size();
}

template <typename Scalar>
unsigned int Polygon<Scalar>::numEdges() const
{
    unsigned int num_edge = 0;
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
        const Group<Scalar> &group = groups_[group_idx];
        num_edge += group.numEdges();
    }
    return num_edge;
}

template <typename Scalar>
unsigned int Polygon<Scalar>::numNormals() const
{
    return vertex_normals_.size();
}

template <typename Scalar>
unsigned int Polygon<Scalar>::numTextureCoordinates() const
{
    return vertex_textures_.size();
}

template <typename Scalar>
unsigned int Polygon<Scalar>::numGroups() const
{
    return groups_.size();
}

template <typename Scalar>
unsigned int Polygon<Scalar>::numMaterials() const
{
    return materials_.size();
}


template <typename Scalar>
unsigned int Polygon<Scalar>::numIsolatedVertices() const
{
    vector<unsigned int> neighor_edge_count(numVertices(),0);
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
        const Group<Scalar> &group = groups_[group_idx];
        for(unsigned int edge_idx = 0; edge_idx < group.numEdges(); ++edge_idx)
        {
            const Edge<Scalar> &edge = group.edge(edge_idx);
            for(unsigned int vert_idx = 0; vert_idx < edge.numVertices(); ++vert_idx)
            {
                const Vertex<Scalar> &vertex = edge.vertex(vert_idx);
                unsigned int pos_idx = vertex.positionIndex();
                ++neighor_edge_count[pos_idx];
            }
        }
    }
    unsigned int num_isolated_vertices = 0;
    for(unsigned int i = 0; i < numVertices(); ++i)
        if(neighor_edge_count[i]>0)
            ++num_isolated_vertices;
    return num_isolated_vertices;
}

template <typename Scalar>
bool Polygon<Scalar>::isTriangularPolygon() const
{
    // to do
    return true;
}

template <typename Scalar>
bool Polygon<Scalar>::isQuadrilateralPolygon() const
{
    // to do
    return true;
}

template <typename Scalar>
const Vector<Scalar,2>& Polygon<Scalar>::vertexPosition(unsigned int vert_idx) const
{
    bool index_valid = (vert_idx>=0)&&(vert_idx<vertex_positions_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_positions_[vert_idx];
}

template <typename Scalar>
const Vector<Scalar,2>& Polygon<Scalar>::vertexPosition(const Vertex<Scalar> &vertex) const
{
    unsigned int vert_idx = vertex.positionIndex();
    return vertexPosition(vert_idx);
}

template <typename Scalar>
void Polygon<Scalar>::setVertexPosition(unsigned int vert_idx, const Vector<Scalar,2> &position)
{
    bool index_valid = (vert_idx>=0)&&(vert_idx<vertex_positions_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon vertex index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_positions_[vert_idx] = position;
}

template <typename Scalar>
void Polygon<Scalar>::setVertexPosition(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &position)
{
    unsigned int vert_idx = vertex.positionIndex();
    setVertexPosition(vert_idx,position);
}

template <typename Scalar>
const Vector<Scalar,2>& Polygon<Scalar>::vertexNormal(unsigned int normal_idx) const
{
    bool index_valid = (normal_idx>=0)&&(normal_idx<vertex_normals_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon vertex normal index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_normals_[normal_idx];
}

template <typename Scalar>
const Vector<Scalar,2>& Polygon<Scalar>::vertexNormal(const Vertex<Scalar> &vertex) const
{
    unsigned int normal_idx = vertex.normalIndex();
    return vertexNormal(normal_idx);
}

template <typename Scalar>
void Polygon<Scalar>::setVertexNormal(unsigned int normal_idx, const Vector<Scalar,2> &normal)
{
    bool index_valid = (normal_idx>=0)&&(normal_idx<vertex_normals_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon vertex normal index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_normals_[normal_idx] = normal;
}

template <typename Scalar>
void Polygon<Scalar>::setVertexNormal(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &normal)
{
    unsigned int normal_idx = vertex.normalIndex();
    setVertexNormal(normal_idx,normal);
}

template <typename Scalar>
const Vector<Scalar,2>& Polygon<Scalar>::vertexTextureCoordinate(unsigned int texture_idx) const
{
    bool index_valid = (texture_idx>=0)&&(texture_idx<vertex_textures_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon vertex texture index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return vertex_textures_[texture_idx];
}

template <typename Scalar>
const Vector<Scalar,2>& Polygon<Scalar>::vertexTextureCoordinate(const Vertex<Scalar> &vertex) const
{
    unsigned int texture_idx = vertex.textureCoordinateIndex();
    return vertexTextureCoordinate(texture_idx);
}

template <typename Scalar>
void Polygon<Scalar>::setVertexTextureCoordinate(unsigned int texture_idx, const Vector<Scalar,2> &texture_coordinate)
{
    bool index_valid = (texture_idx>=0)&&(texture_idx<vertex_textures_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon vertex texture index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    vertex_textures_[texture_idx] = texture_coordinate;
}

template <typename Scalar>
void Polygon<Scalar>::setVertexTextureCoordinate(const Vertex<Scalar> &vertex, const Vector<Scalar,2> &texture_coordinate)
{
    unsigned int texture_idx = vertex.textureCoordinateIndex();
    setVertexTextureCoordinate(texture_idx,texture_coordinate);
}

template <typename Scalar>
const Group<Scalar>& Polygon<Scalar>::group(unsigned int group_idx) const
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return groups_[group_idx];
}

template <typename Scalar>
Group<Scalar>& Polygon<Scalar>::group(unsigned int group_idx)
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return groups_[group_idx];
}

template <typename Scalar>
const Group<Scalar>* Polygon<Scalar>::groupPtr(unsigned int group_idx) const
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return &(groups_[group_idx]);
}

template <typename Scalar>
Group<Scalar>* Polygon<Scalar>::groupPtr(unsigned int group_idx)
{
    bool index_valid = (group_idx>=0)&&(group_idx<groups_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon group index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return &(groups_[group_idx]);
}

template <typename Scalar>
const Group<Scalar>* Polygon<Scalar>::groupPtr(const string &name) const
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
        if(groups_[group_idx].name() == name)
            return &(groups_[group_idx]);
    return NULL;
}

template <typename Scalar>
Group<Scalar>* Polygon<Scalar>::groupPtr(const string &name)
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
        if(groups_[group_idx].name() == name)
            return &(groups_[group_idx]);
    return NULL;
}

template <typename Scalar>
const Material<Scalar>& Polygon<Scalar>::material(unsigned int material_idx) const
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return materials_[material_idx];
}

template <typename Scalar>
Material<Scalar>& Polygon<Scalar>::material(unsigned int material_idx)
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return materials_[material_idx];
} 

template <typename Scalar>
const Material<Scalar>* Polygon<Scalar>::materialPtr(unsigned int material_idx) const
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"Polygon material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return &(materials_[material_idx]);
}

template <typename Scalar>
Material<Scalar>* Polygon<Scalar>::materialPtr(unsigned int material_idx)
{
    bool index_valid = (material_idx>=0)&&(material_idx<materials_.size());
    if(!index_valid)
    {
        std::cerr<<"POlygon material index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    return &(materials_[material_idx]);
}

template <typename Scalar>
unsigned int Polygon<Scalar>::materialIndex(const string &material_name) const
{
    for(unsigned int i = 0; i < materials_.size(); ++i)
        if(materials_[i].name() == material_name)
            return i;
    return -1;
}

template <typename Scalar>
void Polygon<Scalar>::setSingleMaterial(const Material<Scalar> &material)
{
    materials_.clear();
    materials_.push_back(material);
    for(int group_idx = 0; group_idx < groups_.size(); ++group_idx)
        groups_[group_idx].setMaterialIndex(0);
}

template <typename Scalar>
const Edge<Scalar>& Polygon<Scalar>::edge(unsigned int edge_idx) const 
{
    bool index_valid = (edge_idx>=0)&&(edge_idx<numEdges());
    if(!index_valid)
    {
        std::cerr<<"Polygon edge index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_edge_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		const Group<Scalar>& current_group = groups_[group_idx];
		unsigned int group_edge_num = current_group.numEdges();
		if(current_edge_sum + group_edge_num > edge_idx)//find the group containing this edge
		{
			return current_group.edge(edge_idx - current_edge_sum);
		}
		else
			current_edge_sum += group_edge_num;
	}
	std::cerr<<"Polygon edge index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].edge(0);
}

template <typename Scalar>
Edge<Scalar>& Polygon<Scalar>::edge(unsigned int edge_idx) 
{
    bool index_valid = (edge_idx>=0)&&(edge_idx<numEdges());
    if(!index_valid)
    {
        std::cerr<<"Polygon edge index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_edge_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		Group<Scalar>& current_group = groups_[group_idx];
		unsigned int group_edge_num = current_group.numEdges();
		if(current_edge_sum + group_edge_num > edge_idx)//find the group containing this edge
		{
			return current_group.edge(edge_idx - current_edge_sum);
		}
		else
			current_edge_sum += group_edge_num;
	}
	std::cerr<<"Polygon edge index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].edge(0);
}

template <typename Scalar>
const Edge<Scalar>* Polygon<Scalar>::edgePtr(unsigned int edge_idx) const 
{
    bool index_valid = (edge_idx>=0)&&(edge_idx<numEdges());
    if(!index_valid)
    {
        std::cerr<<"Polygon edge index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_edge_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		const Group<Scalar>& current_group = groups_[group_idx];
		unsigned int group_edge_num = current_group.numEdges();
		if(current_edge_sum + group_edge_num > edge_idx)//find the group containing this edge
		{
			return current_group.edgePtr(edge_idx - current_edge_sum);
		}
		else
			current_edge_sum += group_edge_num;
	}
	std::cerr<<"Polygon edge index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].edgePtr(0);
}

template <typename Scalar>
Edge<Scalar>* Polygon<Scalar>::edgePtr(unsigned int edge_idx) 
{
    bool index_valid = (edge_idx>=0)&&(edge_idx<numEdges());
    if(!index_valid)
    {
        std::cerr<<"Polygon edge index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	unsigned int current_edge_sum = 0;
	for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
	{
		Group<Scalar>& current_group = groups_[group_idx];
		unsigned int group_edge_num = current_group.numEdges();
		if(current_edge_sum + group_edge_num > edge_idx)//find the group containing this edge
		{
			return current_group.edgePtr(edge_idx - current_edge_sum);
		}
		else
			current_edge_sum += group_edge_num;
	}
	std::cerr<<"Polygon edge index out of range!\n";
	std::exit(EXIT_FAILURE);
    return groups_[0].edgePtr(0);
}

template <typename Scalar>
void Polygon<Scalar>::addMaterial(const Material<Scalar> &material)
{
    materials_.push_back(material);
}

template <typename Scalar>
void Polygon<Scalar>::addGroup(const Group<Scalar> &group)
{
    groups_.push_back(group);
}

template <typename Scalar>
void Polygon<Scalar>::addVertexPosition(const Vector<Scalar,2> &position)
{
    vertex_positions_.push_back(position);
}

template <typename Scalar>
void Polygon<Scalar>::addVertexNormal(const Vector<Scalar,2> &normal)
{
    vertex_normals_.push_back(normal);
}

template <typename Scalar>
void Polygon<Scalar>::addVertexTextureCoordinate(const Vector<Scalar,2> &texture_coordinate)
{
    vertex_textures_.push_back(texture_coordinate);
}

template <typename Scalar>
void Polygon<Scalar>::computeAllVertexNormals(VertexNormalType normal_type)
{
    switch(normal_type)
    {
    case WEIGHTED_EDGE_NORMAL:
        setVertexNormalsToWeightedEdgeNormals();
        break;
    case AVERAGE_EDGE_NORMAL:
        setVertexNormalsToAverageEdgeNormals();
        break;
    case EDGE_NORMAL:
        setVertexNormalsToEdgeNormals();
        break;
    default:
        std::cerr<<"Wrong normal type specified, use weighted edge nromals!\n";
        setVertexNormalsToWeightedEdgeNormals();
    }
}

template <typename Scalar>
void Polygon<Scalar>::computeAllEdgeNormals()
{
    for(unsigned int group_idx = 0; group_idx < groups_.size(); ++group_idx)
    {
        Group<Scalar> &group = groups_[group_idx];
        for(unsigned int edge_idx = 0; edge_idx < group.numEdges(); ++edge_idx)
        {
            Edge<Scalar> &edge = group.edge(edge_idx);
            computeEdgeNormal(edge);
        }
    }
}

template <typename Scalar>
void Polygon<Scalar>::computeEdgeNormal(Edge<Scalar> &edge)
{
    // to do
}

template <typename Scalar>
void Polygon<Scalar>::setVertexNormalsToEdgeNormals()
{
	// to do
}

template <typename Scalar>
void Polygon<Scalar>::setVertexNormalsToAverageEdgeNormals()
{
	// to do
}

template <typename Scalar>
void Polygon<Scalar>::setVertexNormalsToWeightedEdgeNormals()
{
    // to do
}

//explicit instantitation
template class Polygon<float>;
template class Polygon<double>;

} // end of namespace Physika