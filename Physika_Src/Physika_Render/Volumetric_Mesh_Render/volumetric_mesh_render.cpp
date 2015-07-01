/*
 * @file volumetric_mesh_render.cpp 
 * @Brief render of volumetric mesh.
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
#include <cstddef>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Transform/transform_2d.h"
#include "Physika_Core/Transform/transform_3d.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_render.h"

namespace Physika{

template <typename Scalar, int Dim> const unsigned int VolumetricMeshRender<Scalar, Dim>::render_solid_= 1<<0;
template <typename Scalar, int Dim> const unsigned int VolumetricMeshRender<Scalar, Dim>::render_wireframe_ = 1<<1;
template <typename Scalar, int Dim> const unsigned int VolumetricMeshRender<Scalar, Dim>::render_vertices_ = 1<<2;

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar, Dim>::VolumetricMeshRender()
    :mesh_(NULL),
     transform_(NULL),
     vert_displacement_(NULL),
     enable_displaylist_(true),
     vertex_display_list_id_(0),
     wire_display_list_id_(0),
     solid_display_list_id_(0),
     solid_with_custom_color_vector_display_list_id_(0)
{
    this->initRenderMode();
}

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::VolumetricMeshRender(const VolumetricMesh<Scalar,Dim> *mesh)
    :mesh_(mesh),
     transform_(NULL),
     vert_displacement_(NULL),
     enable_displaylist_(true),
     vertex_display_list_id_(0),
     wire_display_list_id_(0),
     solid_display_list_id_(0),
     solid_with_custom_color_vector_display_list_id_(0)
{
    this->initRenderMode();
}

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::VolumetricMeshRender(const VolumetricMesh<Scalar,Dim> *mesh, const Transform<Scalar, Dim> *transform)
    :mesh_(mesh),
     transform_(transform),
     vert_displacement_(NULL),
     enable_displaylist_(true),
     vertex_display_list_id_(0),
     wire_display_list_id_(0),
     solid_display_list_id_(0),
     solid_with_custom_color_vector_display_list_id_(0)
{
    this->initRenderMode();
}

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::VolumetricMeshRender(const VolumetricMesh<Scalar,Dim> *mesh, const std::vector<Vector<Scalar,Dim> > *vertex_displacement)
    :mesh_(mesh),
     transform_(NULL),
     vert_displacement_(vertex_displacement),
     enable_displaylist_(true),
     vertex_display_list_id_(0),
     wire_display_list_id_(0),
     solid_display_list_id_(0),
     solid_with_custom_color_vector_display_list_id_(0)
{
    this->initRenderMode();
}
    
template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::~VolumetricMeshRender()
{
    this->deleteDisplayLists();
}

template <typename Scalar, int Dim>
const VolumetricMesh<Scalar,Dim>* VolumetricMeshRender<Scalar,Dim>::mesh()const
{
    return this->mesh_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setVolumetricMesh(const VolumetricMesh<Scalar, Dim> *mesh)
{
    this->mesh_ = mesh;
    this->deleteDisplayLists();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setVolumetricMesh(const VolumetricMesh<Scalar, Dim> *mesh, const Transform<Scalar, Dim> *transform)
{
    this->mesh_ = mesh;
    this->transform_ = transform;
    this->deleteDisplayLists();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setVolumetricMesh(const VolumetricMesh<Scalar, Dim> *mesh, const std::vector<Vector<Scalar,Dim> > *vertex_displacement)
{
    this->mesh_ = mesh;
    this->vert_displacement_ = vertex_displacement;
    this->deleteDisplayLists();
}
    
template <typename Scalar, int Dim>
const Transform<Scalar, Dim>* VolumetricMeshRender<Scalar,Dim>::transform()const
{
    return this->transform_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setTransform(const Transform<Scalar, Dim> *transform)
{
    this->transform_ = transform;
}
    
template <typename Scalar, int Dim>
const std::vector<Vector<Scalar,Dim> >* VolumetricMeshRender<Scalar,Dim>::vertexDisplacement()const
{
    return this->vert_displacement_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::setVertexDisplacement(const std::vector<Vector<Scalar,Dim> > *vertex_displacement)
{
    this->vert_displacement_ = vertex_displacement;
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
void VolumetricMeshRender<Scalar,Dim>::enableDisplayList()
{
    enable_displaylist_ = true;
    this->deleteDisplayLists();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::disableDisplayList()
{
    enable_displaylist_ = false;
    this->deleteDisplayLists();
}
    
template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::synchronize()
{
    //for now, synchronize() only calls deleteDisplayLists() internally
    //the reason for defining synchronize() is to hide impmentation detail
    //and provide a more intuitive name for callers 
    deleteDisplayLists();
}

template<typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::printInfo()const
{
	std::cout<<"mesh_address: "<<this->mesh_<<std::endl;
	std::cout<<"transform_address: "<<this->transform_<<std::endl;

	unsigned int render_mode = this->render_mode_;
	std::cout<<"render_mode: ";
	if(render_mode & this->render_solid_)
		std::cout<<"solid ";
	if(render_mode & this->render_wireframe_)
		std::cout<<"wireFrame ";
	if(render_mode & this->render_vertices_)
		std::cout<<"vertex ";
	std::cout<<std::endl;

	std::cout<<"vertex_display_list_id_: "<<this->vertex_display_list_id_<<std::endl;
	std::cout<<"wire_display_list_id_: "<<this->wire_display_list_id_<<std::endl;
	std::cout<<"solid_display_list_id_: "<<this->solid_display_list_id_<<std::endl;
	std::cout<<"solid_with_custom_color_vector_display_list_id_: "<<this->solid_with_custom_color_vector_display_list_id_<<std::endl;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::render()
{
    if(this->mesh_ == NULL)
    {
        std::cerr<<"No mesh is binded to the MeshRender!\n";
        return;
    }
    if(render_mode_ & render_solid_)
        renderSolidWithAlpha();
    if(render_mode_ & render_wireframe_)
        renderWireframe();
    if(render_mode_ & render_vertices_)
        renderVertices();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::renderSolidWithAlpha(float alpha)
{
    PHYSIKA_ASSERT(this->mesh_);
    
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_DEPTH_BUFFER_BIT);

    glDisable(GL_LIGHTING);           // disable lighting
    glDisable(GL_DEPTH_TEST);         // disable depth test
    glEnable(GL_BLEND);               // enable blend
    glBlendFunc(GL_SRC_ALPHA,GL_ONE); // set blend function
    glCullFace(GL_BACK);              // cull back face

    float current_rgba[4];
    glGetFloatv(GL_CURRENT_COLOR, current_rgba);
    current_rgba[3] = alpha;          // set the alpha value

    Color<float> RGBA(current_rgba[0], current_rgba[1], current_rgba[2], current_rgba[3]);
    openGLColor4(RGBA);
    glPushMatrix();
    if(this->transform_ != NULL)
        openGLMultMatrix(this->transform_->transformMatrix());	// add transform

    VolumetricMeshInternal::ElementType ele_type = this->mesh_->elementType();
    if(!glIsList(this->solid_display_list_id_))
    {
        if (this->enable_displaylist_)
        {
            this->solid_display_list_id_ = glGenLists(1);
            glNewList(this->solid_display_list_id_, GL_COMPILE_AND_EXECUTE);
        }

        unsigned int num_ele = this->mesh_->eleNum();
        for(unsigned int ele_idx=0; ele_idx<num_ele; ele_idx++)
        {
            switch(ele_type)
            {
            case VolumetricMeshInternal::TRI:
            case VolumetricMeshInternal::QUAD:
                this->drawTriOrQuad(ele_idx);
                break;
            case VolumetricMeshInternal::TET:
                this->drawTet(ele_idx);
                break;
            case VolumetricMeshInternal::CUBIC:
                this->drawCubic(ele_idx);
                break;
            default:   
                break;
            }
        }
        if (this->enable_displaylist_)
            glEndList();
    }
    else
        glCallList(this->solid_display_list_id_);

    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim> template <typename ColorType>
void VolumetricMeshRender<Scalar,Dim>::renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const Color<ColorType>  &color)
{
    PHYSIKA_ASSERT(this->mesh_);
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_POINT);             // enable polygon offset
    glPolygonOffset(-1.0,1.0); 
    openGLColor3(color);
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(static_cast<float>(1.5*point_size));

    glPushMatrix();
    if(this->transform_ != NULL)
        openGLMultMatrix(this->transform_->transformMatrix());	
  
    unsigned int num_vertex = vertex_id.size();
    glBegin(GL_POINTS);
    for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
    {
        Vector<Scalar,Dim> position = this->mesh_->vertPos(vertex_id[vertex_idx]);
        if(vert_displacement_) //apply vertex displacement if any
            position += (*vert_displacement_)[vertex_idx];
        openGLVertex(position);
    }
    glEnd();
   
    glPopMatrix();
    glPopAttrib();

}

template <typename Scalar, int Dim> template <typename ColorType>
void VolumetricMeshRender<Scalar,Dim>::renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const std::vector< Color<ColorType> > &color)
{
    PHYSIKA_ASSERT(this->mesh_);
    if(vertex_id.size()!= color.size())
        std::cerr<<"Warning: vertex number doesn't match color number, white used as default color!\n";
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_POINT);             // enable polygon offset
    glPolygonOffset(-1.0,1.0);
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(static_cast<float>(1.5*point_size));

    glPushMatrix();
    if(this->transform_ != NULL)
        openGLMultMatrix(this->transform_->transformMatrix());
   
    unsigned int num_vertex = vertex_id.size();
    glBegin(GL_POINTS);
    for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
    {
        if(vertex_idx<color.size())
            openGLColor3(color[vertex_idx]);
        else
            openGLColor3(Color<ColorType>::White());
        Vector<Scalar,Dim> position = this->mesh_->vertPos(vertex_id[vertex_idx]);
        if(vert_displacement_) //apply vertex displacement if any
        {
            if(vert_displacement_->size() != num_vertex)
                throw PhysikaException("Vertex displacement and vertex number mismatch!");
            position += (*vert_displacement_)[vertex_idx];
        }
        openGLVertex(position);
    }
    glEnd();
    
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim> template <typename ColorType>
void VolumetricMeshRender<Scalar,Dim>::renderElementWithColor(const std::vector<unsigned int> &element_id, const Color<ColorType>  &color)
{
    PHYSIKA_ASSERT(this->mesh_);
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glPolygonMode(GL_FRONT, GL_FILL);              // set polygon mode FILL for SOLID MODE
    glCullFace(GL_BACK);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_FILL);              // enable polygon offset
    openGLColor3(color);
    glPolygonOffset(-1.0,1.0);                      // set polygon offset (factor, unit)

    glPushMatrix();
    if(this->transform_ != NULL)
        openGLMultMatrix(this->transform_->transformMatrix());
    VolumetricMeshInternal::ElementType ele_type = this->mesh_->elementType();

    unsigned int num_face = element_id.size();     
    for(unsigned int ele_idx=0; ele_idx<num_face; ele_idx++)
    {
        switch(ele_type)
        {
        case VolumetricMeshInternal::TRI:
        case VolumetricMeshInternal::QUAD:
            this->drawTriOrQuad(element_id[ele_idx]);
            break;
        case VolumetricMeshInternal::TET:   
            this->drawTet(element_id[ele_idx]);
            break;
        case VolumetricMeshInternal::CUBIC:
            this->drawCubic(element_id[ele_idx]);
            break;
        default:
            break;
        }
    }
   
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim> template <typename ColorType>
void VolumetricMeshRender<Scalar,Dim>::renderElementWithColor(const std::vector<unsigned int> &element_id, const std::vector< Color<ColorType> > &color)
{
    if(element_id.size()!= color.size())
        std::cerr<<"warning: the size of element_id don't equal to color's, the elements lacking of cunstom color will be rendered in white color !!"<<std::endl;
    
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glPolygonMode(GL_FRONT, GL_FILL);              // set polygon mode FILL for SOLID MODE
    glCullFace(GL_BACK);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_FILL);              // enable polygon offset
    glPolygonOffset(-1.0,1.0);                      // set polygon offset (factor, unit)

    glPushMatrix();
    if(this->transform_ != NULL)
        openGLMultMatrix(this->transform_->transformMatrix());
    
    VolumetricMeshInternal::ElementType ele_type = this->mesh_->elementType();
   
    unsigned int num_face = element_id.size();     
    for(unsigned int ele_idx=0; ele_idx<num_face; ele_idx++)
    {
        if(ele_idx<color.size())
            openGLColor3(color[ele_idx]);
        else
            openGLColor3(Color<ColorType>::White());

        switch(ele_type)
        {
        case VolumetricMeshInternal::TRI:
        case VolumetricMeshInternal::QUAD:
            this->drawTriOrQuad(element_id[ele_idx]);
            break;
        case VolumetricMeshInternal::TET:   
            this->drawTet(element_id[ele_idx]);
            break;
        case VolumetricMeshInternal::CUBIC:
            this->drawCubic(element_id[ele_idx]);
            break;
        default:
            break;
        }
    }
    
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim> template <typename ColorType>
void VolumetricMeshRender<Scalar,Dim>::renderSolidWithCustomColor(const std::vector< Color<ColorType> > &color)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::initRenderMode()
{
    //default render mode: solid
    render_mode_ = 0;
    render_mode_ |= render_solid_;
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::renderVertices()
{
    PHYSIKA_ASSERT(this->mesh_);
    
    glPushAttrib(GL_LIGHTING_BIT|GL_ENABLE_BIT);
    glPushMatrix();
    if(this->transform_ != NULL)
        openGLMultMatrix(this->transform_->transformMatrix());
    if(!glIsList(this->vertex_display_list_id_))
    {
        if (this->enable_displaylist_)
        {
            this->vertex_display_list_id_ = glGenLists(1);
            glNewList(this->vertex_display_list_id_, GL_COMPILE_AND_EXECUTE);
        }
        
        glDisable(GL_LIGHTING);
        unsigned int num_vertex = this->mesh_->vertNum();
        glBegin(GL_POINTS);
        for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
        {
            Vector<Scalar,Dim> position = this->mesh_->vertPos(vertex_idx);
            if(vert_displacement_) //apply vertex displacement if any
            {
                if(vert_displacement_->size() != num_vertex)
                    throw PhysikaException("Vertex displacement and vertex number mismatch!");
                position += (*vert_displacement_)[vertex_idx];
            }
            openGLVertex(position);
        }
        glEnd();
        if (this->enable_displaylist_)
            glEndList();
    }
    else
        glCallList(this->vertex_display_list_id_);
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::renderWireframe()
{
    PHYSIKA_ASSERT(this->mesh_);
    
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);        // set openGL polygon mode for wire mode
    glDisable(GL_LIGHTING);                           // disable lighting

    glPushMatrix();
    if(this->transform_ != NULL)
        openGLMultMatrix(this->transform_->transformMatrix());	

    VolumetricMeshInternal::ElementType ele_type = this->mesh_->elementType();
    if(!glIsList(this->wire_display_list_id_))
    {
        if (this->enable_displaylist_)
        {
            this->wire_display_list_id_ = glGenLists(1);
            glNewList(this->wire_display_list_id_, GL_COMPILE_AND_EXECUTE);
        }
        unsigned int num_ele = this->mesh_->eleNum();
        for(unsigned int ele_idx=0; ele_idx<num_ele; ele_idx++)
        {
            switch(ele_type)
            {
            case VolumetricMeshInternal::TRI:
            case VolumetricMeshInternal::QUAD:
                this->drawTriOrQuad(ele_idx);
                break;
            case VolumetricMeshInternal::TET:
                this->drawTet(ele_idx);
                break;
            case VolumetricMeshInternal::CUBIC:
                this->drawCubic(ele_idx);
                break;
            default:
                break;
            }
        }
        if (this->enable_displaylist_)
            glEndList();
    }
    else
        glCallList(this->wire_display_list_id_);
    glPopMatrix();
    glPopAttrib();
}
    
template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::drawTriOrQuad(unsigned int ele_idx)
{
    PHYSIKA_ASSERT(this->mesh_);
    PHYSIKA_ASSERT(ele_idx < this->mesh_->eleNum());
    unsigned int ele_vert_num = this->mesh_->eleVertNum(ele_idx);
    glBegin(GL_POLYGON);
    for(unsigned int vert_idx = 0; vert_idx < ele_vert_num; vert_idx++)
    {
        Vector<Scalar,Dim> position = this->mesh_->eleVertPos(ele_idx,vert_idx);
        if(vert_displacement_)
        {
            if(vert_displacement_->size() != this->mesh_->vertNum())
                throw PhysikaException("Vertex displacement and vertex number mismatch!");
            unsigned int global_vert_idx = this->mesh_->eleVertIndex(ele_idx,vert_idx);
            position += (*vert_displacement_)[global_vert_idx];
        }
        openGLVertex(position);
    }
    glEnd();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::drawTet(unsigned int ele_idx)
{
    PHYSIKA_ASSERT(this->mesh_);
    PHYSIKA_ASSERT(ele_idx < this->mesh_->eleNum());
    
    Vector<Scalar,Dim> position_0 = this->mesh_->eleVertPos(ele_idx,0);
    Vector<Scalar,Dim> position_1 = this->mesh_->eleVertPos(ele_idx,1);
    Vector<Scalar,Dim> position_2 = this->mesh_->eleVertPos(ele_idx,2);
    Vector<Scalar,Dim> position_3 = this->mesh_->eleVertPos(ele_idx,3);
    //apply displacment if any
    if(vert_displacement_)
    {
        if(vert_displacement_->size() != this->mesh_->vertNum())
            throw PhysikaException("Vertex displacement and vertex number mismatch!");
        position_0 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,0)];
        position_1 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,1)];
        position_2 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,2)];
        position_3 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,3)];
    }

    glBegin(GL_POLYGON);
        openGLVertex(position_0);
        openGLVertex(position_2);
        openGLVertex(position_1);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_0);
        openGLVertex(position_1);
        openGLVertex(position_3);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_1);
        openGLVertex(position_2);
        openGLVertex(position_3);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_0);
        openGLVertex(position_3);
        openGLVertex(position_2);
    glEnd();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::drawCubic(unsigned int ele_idx)
{
    PHYSIKA_ASSERT(this->mesh_);
    PHYSIKA_ASSERT(ele_idx < this->mesh_->eleNum());
    
    Vector<Scalar,Dim> position_0 = this->mesh_->eleVertPos(ele_idx,0);
    Vector<Scalar,Dim> position_1 = this->mesh_->eleVertPos(ele_idx,1);
    Vector<Scalar,Dim> position_2 = this->mesh_->eleVertPos(ele_idx,2);
    Vector<Scalar,Dim> position_3 = this->mesh_->eleVertPos(ele_idx,3);
    Vector<Scalar,Dim> position_4 = this->mesh_->eleVertPos(ele_idx,4);
    Vector<Scalar,Dim> position_5 = this->mesh_->eleVertPos(ele_idx,5);
    Vector<Scalar,Dim> position_6 = this->mesh_->eleVertPos(ele_idx,6);
    Vector<Scalar,Dim> position_7 = this->mesh_->eleVertPos(ele_idx,7);
    //apply displacement if any
    if(vert_displacement_)
    {
        if(vert_displacement_->size() != this->mesh_->vertNum())
            throw PhysikaException("Vertex displacement and vertex number mismatch!");
        position_0 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,0)];
        position_1 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,1)];
        position_2 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,2)];
        position_3 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,3)];
        position_4 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,4)];
        position_5 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,5)];
        position_6 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,6)];
        position_7 += (*vert_displacement_)[this->mesh_->eleVertIndex(ele_idx,7)];
    }
    
    glBegin(GL_POLYGON);
        openGLVertex(position_0);
        openGLVertex(position_3);
        openGLVertex(position_2);
        openGLVertex(position_1);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_4);
        openGLVertex(position_5);
        openGLVertex(position_6);
        openGLVertex(position_7);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_0);
        openGLVertex(position_1);
        openGLVertex(position_5);
        openGLVertex(position_4);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_3);
        openGLVertex(position_7);
        openGLVertex(position_6);
        openGLVertex(position_2);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_4);
        openGLVertex(position_7);
        openGLVertex(position_3);
        openGLVertex(position_0);
    glEnd();

    glBegin(GL_POLYGON);
        openGLVertex(position_1);
        openGLVertex(position_2);
        openGLVertex(position_6);
        openGLVertex(position_5);
    glEnd();
}

template <typename Scalar, int Dim>
void VolumetricMeshRender<Scalar,Dim>::deleteDisplayLists()
{
    //old displaylists are deleted whenever synchronization is needed
    glDeleteLists(this->solid_display_list_id_, 1);
    glDeleteLists(this->wire_display_list_id_, 1);
    glDeleteLists(this->vertex_display_list_id_, 1);
    glDeleteLists(this->solid_with_custom_color_vector_display_list_id_, 1);

    this->solid_display_list_id_ = 0;
    this->wire_display_list_id_ = 0;
    this->vertex_display_list_id_ = 0;
    this->solid_with_custom_color_vector_display_list_id_ = 0;
}

// explicit instantitation
template class VolumetricMeshRender<float,2> ;
template class VolumetricMeshRender<double,2>;
template class VolumetricMeshRender<float,3>;
template class VolumetricMeshRender<double,3>;

//for each color type: char,short,int,float,double,unsigned char,unsigned short,unsigned int
//explicit instantiate the render**WithColor Method

// renderElementWithColor
template void VolumetricMeshRender<float,2>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);

template void VolumetricMeshRender<float,3>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);

template void VolumetricMeshRender<float,2>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<float,2>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<double,2>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);

template void VolumetricMeshRender<float,3>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<float,3>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<double,3>::renderElementWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);

//renderVertexWithColor
template void VolumetricMeshRender<float,2>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);

template void VolumetricMeshRender<float,3>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);


template void VolumetricMeshRender<float,2>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<float,2>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<double,2>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);

template void VolumetricMeshRender<float,3>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<float,3>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short>>&);
template void VolumetricMeshRender<double,3>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int>>&);

}  //end of namespace Physika
