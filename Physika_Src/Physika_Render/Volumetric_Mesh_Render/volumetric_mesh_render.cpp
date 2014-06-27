/*
 * @file volumetric_mesh_render.cpp 
 * @Brief render of volumetric mesh.
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
#include <iostream>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Core/Transform/transform.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_render.h"

namespace Physika{

template <typename Scalar, int Dim> const unsigned int VolumetricMeshRender<Scalar, Dim>::render_solid_= 1<<0;
template <typename Scalar, int Dim> const unsigned int VolumetricMeshRender<Scalar, Dim>::render_wireframe_ = 1<<1;
template <typename Scalar, int Dim> const unsigned int VolumetricMeshRender<Scalar, Dim>::render_vertices_ = 1<<2;

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar, Dim>::VolumetricMeshRender()
    :mesh_(NULL),
	transform_(NULL),
	solid_display_list_id_(0),
    wire_display_list_id_(0),
	vertex_display_list_id_(0),
	element_with_color_display_list_id_(0),
	element_with_color_vector_display_list_id_(0),
	vertex_with_color_display_list_id_(0),
	vertex_with_color_vector_display_list_id_(0),
	solid_with_custom_color_vector_display_list_id_(0)
{
	this->initRenderMode();
}

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::VolumetricMeshRender(VolumetricMesh<Scalar,Dim>* mesh)
	:mesh_(mesh),
	transform_(NULL),
	solid_display_list_id_(0),
    wire_display_list_id_(0),
	vertex_display_list_id_(0),
	element_with_color_display_list_id_(0),
	element_with_color_vector_display_list_id_(0),
	vertex_with_color_display_list_id_(0),
	vertex_with_color_vector_display_list_id_(0),
	solid_with_custom_color_vector_display_list_id_(0)
{
	this->initRenderMode();
}

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::VolumetricMeshRender(VolumetricMesh<Scalar,Dim>* mesh, Transform<Scalar>* transform)
	:mesh_(mesh),
	transform_(transform),
	solid_display_list_id_(0),
    wire_display_list_id_(0),
	vertex_display_list_id_(0),
	element_with_color_display_list_id_(0),
	element_with_color_vector_display_list_id_(0),
	vertex_with_color_display_list_id_(0),
	vertex_with_color_vector_display_list_id_(0),
	solid_with_custom_color_vector_display_list_id_(0)
{
	this->initRenderMode();
}

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::~VolumetricMeshRender()
{
	this->transform_ = NULL;
	this->deleteDisplayLists();
}

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar,Dim>* VolumetricMeshRender<Scalar,Dim>::mesh()const
{
	return this->mesh_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setVolumetricMesh(VolumetricMesh<Scalar, Dim>* mesh)
{
	this->mesh_ = mesh;
	this->deleteDisplayLists();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setVolumetricMesh(VolumetricMesh<Scalar, Dim>* mesh, Transform<Scalar>*transform)
{
	this->mesh_ = mesh;
	this->transform_ = transform;
	this->deleteDisplayLists();
}

template <typename Scalar, int Dim>
const Transform<Scalar>* VolumetricMeshRender<Scalar,Dim>::transform()const
{
	return this->transform_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setTransform(Transform<Scalar>* transform)
{
	this->transform_ = transform;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::enableRenderSolid()
{
    render_mode_ |= render_solid_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::disableRenderSolid()
{
    render_mode_ &= ~render_solid_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::enableRenderVertices()
{
    render_mode_ |= render_vertices_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::disableRenderVertices()
{
    render_mode_ &= ~render_vertices_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::enableRenderWireframe()
{
    render_mode_ |= render_wireframe_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::disableRenderWireframe()
{
    render_mode_ &= ~render_wireframe_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::synchronize()
{
    //for now, synchronize() only calls deleteDisplayLists() internally
    //the reason for defining synchronize() is to hide impmentation detail
    //and provide a more intuitive name for callers 
    deleteDisplayLists();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::render()
{
    if(this->mesh_==NULL)
    {
        std::cerr<<"No mesh is binded to the MeshRender!\n";
        return;
    }
    if(render_mode_ & render_solid_)
        renderSolid();
    if(render_mode_ & render_wireframe_)
        renderWireframe();
    if(render_mode_ & render_vertices_)
        renderVertices();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::initRenderMode()
{
    //default render mode: solid
    render_mode_ = 0;
    render_mode_ |= render_solid_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::deleteDisplayLists()
{
    //old displaylists are deleted whenever synchronization is needed
    glDeleteLists(this->solid_display_list_id_, 1);
    glDeleteLists(this->wire_display_list_id_, 1);
    glDeleteLists(this->vertex_display_list_id_, 1);
	glDeleteLists(this->element_with_color_display_list_id_, 1);
	glDeleteLists(this->element_with_color_vector_display_list_id_, 1);
	glDeleteLists(this->vertex_with_color_display_list_id_, 1);
	glDeleteLists(this->vertex_with_color_vector_display_list_id_, 1);
	glDeleteLists(this->solid_with_custom_color_vector_display_list_id_, 1);

    this->solid_display_list_id_ = 0;
    this->wire_display_list_id_ = 0;
    this->vertex_display_list_id_ = 0;
	this->element_with_color_display_list_id_ = 0;
	this->element_with_color_vector_display_list_id_ = 0;
	this->vertex_with_color_display_list_id_ = 0;
	this->vertex_with_color_vector_display_list_id_ = 0;
	this->solid_with_custom_color_vector_display_list_id_ = 0;
}

}  //end of namespace Physika
