/*
 * @file surface_mesh_render.cpp 
 * @Basic render of surface mesh.
 * @author Fei Zhu ,Wei Chen
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
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_IO/Image_IO/image_io.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Core/Transform/transform.h"
//#include "Physika_Core/Image/image.h"


namespace Physika{

//init render mode flags
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_solid_ = 1<<0;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_wireframe_ = 1<<1;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_vertices_ = 1<<2;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_flat_or_smooth_ = 1<<3;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_texture_ = 1<<5;
    
template <typename Scalar>
SurfaceMeshRender<Scalar>::SurfaceMeshRender()
    :mesh_(NULL),
    transform_(NULL),
    solid_display_list_id_(0),
    wire_display_list_id_(0),
    vertex_display_list_id_(0),
    solid_with_custom_color_vector_display_list_id_(0)
{
    initRenderMode();
}

template <typename Scalar>
SurfaceMeshRender<Scalar>::SurfaceMeshRender(SurfaceMesh<Scalar>* mesh)
    :mesh_(mesh),
    transform_(NULL),
    solid_display_list_id_(0),
    wire_display_list_id_(0),
    vertex_display_list_id_(0),
    solid_with_custom_color_vector_display_list_id_(0)
{
    initRenderMode();
    loadTextures();
}

template <typename Scalar>
SurfaceMeshRender<Scalar>::SurfaceMeshRender(SurfaceMesh<Scalar>* mesh, Transform<Scalar>* transform)
    :mesh_(mesh),
    transform_(transform),
    solid_display_list_id_(0),
    wire_display_list_id_(0),
    vertex_display_list_id_(0),
    solid_with_custom_color_vector_display_list_id_(0)
{
    initRenderMode();
    loadTextures();
}

template <typename Scalar>
SurfaceMeshRender<Scalar>::~SurfaceMeshRender()
{
    releaseTextures();
    deleteDisplayLists();
    transform_ = NULL;
}

template <typename Scalar>
const SurfaceMesh<Scalar>* SurfaceMeshRender<Scalar>::mesh() const
{
    return mesh_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::setSurfaceMesh(SurfaceMesh<Scalar> *mesh)
{
    mesh_ = mesh;
    //after updating the mesh, the textures needed to be update correspondently
    releaseTextures();
    loadTextures();
    deleteDisplayLists();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::setSurfaceMesh(SurfaceMesh<Scalar> *mesh, Transform<Scalar> *transform)
{
    this->setSurfaceMesh(mesh);
    transform_ = transform;
}

template <typename Scalar>
const Transform<Scalar>* SurfaceMeshRender<Scalar>::transform()const
{
    return this->transform_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::setTransform(Transform<Scalar>* transform)
{
    this->transform_ = transform;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableRenderSolid()
{
    render_mode_ |= render_solid_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableRenderSolid()
{
    render_mode_ &= ~render_solid_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableRenderVertices()
{
    render_mode_ |= render_vertices_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableRenderVertices()
{
    render_mode_ &= ~render_vertices_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableRenderWireframe()
{
    render_mode_ |= render_wireframe_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableRenderWireframe()
{
    render_mode_ &= ~render_wireframe_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableFlatShading()
{
    render_mode_ &= ~render_flat_or_smooth_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableSmoothShading()
{
    render_mode_ |= render_flat_or_smooth_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableTexture()
{
    render_mode_ |= render_texture_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableTexture()
{
    render_mode_ &= ~render_texture_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::synchronize()
{
    //for now, synchronize() only calls deleteDisplayLists() internally
    //the reason for defining synchronize() is to hide impmentation detail
    //and provide a more intuitive name for callers 
    deleteDisplayLists();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::render()
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

template<typename Scalar>
void SurfaceMeshRender<Scalar>::printInfo()const
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
	if(render_mode & this->render_texture_)
		std::cout<<"texture ";
	if(render_mode & this->render_flat_or_smooth_)
		std::cout<<"smooth ";
	else
		std::cout<<"flat ";
	std::cout<<std::endl;
	std::cout<<"texture: "<<this->textures_.size()<<" in all, ";
	unsigned int texture_num_available = 0;
	for(unsigned int i=0; i<this->textures_.size(); i++)
	{
		if(this->textures_[i].first == true)
			texture_num_available++;
	}
	std::cout<<texture_num_available<<" available."<<std::endl;
	std::cout<<"vertex_display_list_id_: "<<this->vertex_display_list_id_<<std::endl;
	std::cout<<"wire_display_list_id_: "<<this->wire_display_list_id_<<std::endl;
	std::cout<<"solid_display_list_id_: "<<this->solid_display_list_id_<<std::endl;
	std::cout<<"solid_with_custom_color_vector_display_list_id_: "<<this->solid_with_custom_color_vector_display_list_id_<<std::endl;

}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::initRenderMode()
{
    //default render mode: solid, smooth shading, texture
    render_mode_ = 0;
    render_mode_ |= render_solid_;
    render_mode_ |= render_flat_or_smooth_;
    render_mode_ |= render_texture_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderVertices()
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT);

    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());	
    }
    if(! glIsList(this->vertex_display_list_id_))
    {
        this->vertex_display_list_id_=glGenLists(1);
        glNewList(this->vertex_display_list_id_, GL_COMPILE_AND_EXECUTE);
        glDisable(GL_LIGHTING);
        unsigned int num_vertex = this->mesh_->numVertices();      //get the number of vertices
        glBegin(GL_POINTS);                                        //draw points
        for(unsigned int i=0; i<num_vertex; i++)
        {
            Vector<Scalar,3> position = this->mesh_->vertexPosition(i);
            openGLVertex(position);
        }
        glEnd();
        glEndList();
    }
    else
    {
        glCallList(this->vertex_display_list_id_);
    }
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderWireframe()
{ 
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);        // set openGL polygon mode for wire mode
    glDisable(GL_LIGHTING);

    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());	
    }
    if(! glIsList(this->wire_display_list_id_))
    {
        this->wire_display_list_id_=glGenLists(1);
        glNewList(this->wire_display_list_id_, GL_COMPILE_AND_EXECUTE);
        unsigned int num_group = this->mesh_->numGroups();                 // get group number
        for(unsigned int group_idx=0; group_idx<num_group; group_idx++)    // loop for every group
        {
            Group<Scalar> group_ref = this->mesh_->group(group_idx);       // get group reference
            unsigned int num_face = group_ref.numFaces();                  // get face number
            for(unsigned int face_idx=0; face_idx<num_face; face_idx++)    // loop for every face
            {
                Face<Scalar> face_ref = group_ref.face(face_idx);          // get face reference
                unsigned int num_vertex = face_ref.numVertices();          // get vertex number of face
                glBegin(GL_POLYGON);                                       // draw polygon with wire mode
                for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
                {
                    unsigned position_ID = face_ref.vertex(vertex_idx).positionIndex();   // get vertex positionIndex in "surface mesh"
                    Vector<Scalar,3> position = this->mesh_->vertexPosition(position_ID); // get the position of vertex which is stored in "surface mesh"
                    openGLVertex(position);
                }
                glEnd();
            }
        }
        glEndList();
    }
    else
    {
        glCallList(this->wire_display_list_id_);
    }
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderSolid()
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // set polygon mode FILL for SOLID MODE
    glDisable(GL_COLOR_MATERIAL);              /// warning: we have to disable GL_COLOR_MATERIAL, otherwise the material propertity won't appear!!!
    glEnable(GL_LIGHTING);                     
   
    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());	
    }
    if (! glIsList(this->solid_display_list_id_))
    {   
        this->solid_display_list_id_=glGenLists(1);

        glNewList(this->solid_display_list_id_, GL_COMPILE_AND_EXECUTE);
        unsigned int num_group = this->mesh_->numGroups();                 // get group number
        for(unsigned int group_idx=0; group_idx<num_group; group_idx++)    // loop for every group
        {
            Group<Scalar> group_ref = this->mesh_->group(group_idx);       // get group reference
            unsigned int num_face = group_ref.numFaces();                  // get face number
            unsigned int material_ID = group_ref.materialIndex();

            // get material propety from mesh according to its materialIndex
            Vector<Scalar,3> Ka = this->mesh_->material(material_ID).Ka();
            Vector<Scalar,3> Kd = this->mesh_->material(material_ID).Kd();
            Vector<Scalar,3> Ks = this->mesh_->material(material_ID).Ks();
            Scalar    shininess = this->mesh_->material(material_ID).shininess();
            Scalar        alpha = this->mesh_->material(material_ID).alpha();

            
            GLfloat ambient[4]  = { static_cast<GLfloat>(Ka[0]), static_cast<GLfloat>(Ka[1]), static_cast<GLfloat>(Ka[2]), static_cast<GLfloat>(alpha) };
            GLfloat diffuse[4]  = { static_cast<GLfloat>(Kd[0]), static_cast<GLfloat>(Kd[1]), static_cast<GLfloat>(Kd[2]), static_cast<GLfloat>(alpha) };
            GLfloat specular[4] = { static_cast<GLfloat>(Ks[0]), static_cast<GLfloat>(Ks[1]), static_cast<GLfloat>(Ks[2]), static_cast<GLfloat>(alpha) };
            
            // set material propety for group
            openGLMaterialv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
            openGLMaterialv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
            openGLMaterialv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
            openGLMaterial (GL_FRONT_AND_BACK, GL_SHININESS, static_cast<GLfloat>(shininess));

            // if have a texture, then enable it
            if(this->textures_[material_ID].first==true && (this->render_mode_ & render_texture_) )
            {
                glEnable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D,this->textures_[material_ID].second);
            }
            else
            {
                glDisable(GL_TEXTURE_2D);
            }

            for(unsigned int face_idx=0; face_idx<num_face; face_idx++)    // loop for every face
            {
                Face<Scalar> face_ref = group_ref.face(face_idx);          // get face reference
                unsigned int num_vertex = face_ref.numVertices();          // get vertex number of face
                glBegin(GL_POLYGON);                                       // draw polygon with SOLID MODE
                for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++) // loop for every vertex
                {
                    // if use smooth mode 
                    if(render_mode_ & render_flat_or_smooth_)
                    {
                        if(face_ref.vertex(vertex_idx).hasNormal())
                        {
                            unsigned int vertex_normal_ID = face_ref.vertex(vertex_idx).normalIndex();
                            Vector<Scalar,3> vertex_normal = this->mesh_->vertexNormal(vertex_normal_ID);
                            openGLNormal(vertex_normal);
                        }
                    }
                    else if(face_ref.hasFaceNormal())
                    {
                        Vector<Scalar,3> face_normal = face_ref.faceNormal();
                        openGLNormal(face_normal);

                    }

                    // if vertex has a texture coordinate
                    if(face_ref.vertex(vertex_idx).hasTexture()) 
                    {
                        unsigned int vertex_texture_ID = face_ref.vertex(vertex_idx).textureCoordinateIndex();
                        Vector<Scalar,2> vertex_textureCoord = this->mesh_->vertexTextureCoordinate(vertex_texture_ID);
                        openGLTexCoord(vertex_textureCoord);
                    }
                    unsigned position_ID = face_ref.vertex(vertex_idx).positionIndex();   // get vertex positionIndex in "surface mesh"
                    Vector<Scalar,3> position = this->mesh_->vertexPosition(position_ID); // get the position of vertex which is stored in "surface mesh"
                    openGLVertex(position);
                }
                glEnd();
            }

            glDisable(GL_TEXTURE_2D);
        }
        glEndList();
    }
    else
    {
        glCallList(this->solid_display_list_id_);
    }
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar> template<typename ColorType>
void SurfaceMeshRender<Scalar>::renderSolidWithCustomColor(const std::vector< Color<ColorType> > & color)
{
    if(this->mesh_->numVertices()!= color.size())
    {
        std::cerr<<"warning: the size of color don't equal to vertex number in SurfaceMesh, the vertex lacking of cunstom color will be rendered in white color !"<<std::endl;
    }
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // set polygon mode FILL for SOLID MODE
    glShadeModel(GL_SMOOTH);                   // set shade model to GL_SMOOTH
    glDisable(GL_LIGHTING);                    // disable lighting

    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());	
    }
    if (! glIsList(this->solid_with_custom_color_vector_display_list_id_))
    {   
        this->solid_with_custom_color_vector_display_list_id_=glGenLists(1);
        glNewList(this->solid_with_custom_color_vector_display_list_id_, GL_COMPILE_AND_EXECUTE);

        unsigned int num_group = this->mesh_->numGroups();                 // get group number
        for(unsigned int group_idx=0; group_idx<num_group; group_idx++)    // loop for every group
        {
            Group<Scalar> group_ref = this->mesh_->group(group_idx);       // get group reference
            unsigned int num_face = group_ref.numFaces();                  // get face number
    
            for(unsigned int face_idx=0; face_idx<num_face; face_idx++)    // loop for every face
            {
                Face<Scalar> face_ref = group_ref.face(face_idx);          // get face reference
                unsigned int num_vertex = face_ref.numVertices();          // get vertex number of face
                glBegin(GL_POLYGON);                                       // draw polygon with SOLID MODE
                for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++) // loop for every vertex
                {
                    unsigned position_ID = face_ref.vertex(vertex_idx).positionIndex();   // get vertex positionIndex in "surface mesh"
                    if(position_ID < color.size())
                        openGLColor3(color[position_ID]);
                    else
                        openGLColor3(Color<ColorType>::White());
                    Vector<Scalar,3> position = this->mesh_->vertexPosition(position_ID); // get the position of vertex which is stored in "surface mesh"
                    openGLVertex(position);
                }
                glEnd();
            }
        }
        glEndList();
    }
    else
    {
        glCallList(this->solid_with_custom_color_vector_display_list_id_);
    }
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar> template<typename ColorType>
void SurfaceMeshRender<Scalar>::renderFaceWithColor(const std::vector<unsigned int> &face_id, const Color<ColorType> &color)
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);     // set polygon mode FILL for SOLID MODE
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_FILL);              // enable polygon offset
    openGLColor3(color);
    glPolygonOffset(-1.0,1.0);                     // set polygon offset (factor, unit)

    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());	
    }
  
    unsigned int num_face = face_id.size();     
    for(unsigned int face_idx=0; face_idx<num_face; face_idx++)
    {
        const Face<Scalar>& face_ref = this->mesh_->face(face_id[face_idx]);      //get the reference of face with face_id: face_idx
        unsigned int num_vertex = face_ref.numVertices();
        glBegin(GL_POLYGON);                                                      // draw specific face
        for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
        {
            unsigned position_ID = face_ref.vertex(vertex_idx).positionIndex();   // get vertex positionIndex in "surface mesh"
            Vector<Scalar,3> position = this->mesh_->vertexPosition(position_ID); // get the position of vertex which is stored in "surface mesh"
            openGLVertex(position);
        }
        glEnd();
    }
  
    glPopAttrib();
    glPopMatrix();
}

template <typename Scalar> template<typename ColorType>
void SurfaceMeshRender<Scalar>::renderFaceWithColor(const std::vector<unsigned int> &face_id, const std::vector< Color<ColorType> > &color)
{
    if(face_id.size()!= color.size())
    {
        std::cerr<<"warning: the size of face_id don't equal to color's, the face lacking of cunstom color will be rendered in black color !"<<std::endl;
    }
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);     // set polygon mode FILL for SOLID MODE
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_FILL);              // enable polygon offset
    glPolygonOffset(-1.0,1.0);                      // set polygon offset (factor, unit)

    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());	
    }
   
    unsigned int num_face = face_id.size();     
    for(unsigned int face_idx=0; face_idx<num_face; face_idx++)
    {
        const Face<Scalar>& face_ref = this->mesh_->face(face_id[face_idx]);      //get the reference of face with face_id: face_idx
        unsigned int num_vertex = face_ref.numVertices();
        if(face_idx<color.size())
            openGLColor3(color[face_idx]);
        else
            openGLColor3(Color<ColorType>::White());
        glBegin(GL_POLYGON);                                                      // draw specific face
        for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
        {
            unsigned position_ID = face_ref.vertex(vertex_idx).positionIndex();   // get vertex positionIndex in "surface mesh"
            Vector<Scalar,3> position = this->mesh_->vertexPosition(position_ID); // get the position of vertex which is stored in "surface mesh"
            openGLVertex(position);
        }
        glEnd();
    }
    
    glPopMatrix();
    glPopAttrib();

}

template <typename Scalar> template<typename ColorType>
void SurfaceMeshRender<Scalar>::renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const Color<ColorType> &color)
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_POINT);             // enable polygon offset
    glPolygonOffset(-1.0,1.0); 
    openGLColor3(color);
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(static_cast<float>(.5*point_size));

    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());  
    }
   
    unsigned int num_vertex = vertex_id.size();
    glBegin(GL_POINTS);
    for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
    {
        Vector<Scalar,3> position = this->mesh_->vertexPosition(vertex_id[vertex_idx]); // get the position of vertex which is stored in "surface mesh"
        openGLVertex(position);
    }
    glEnd();
   
    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar> template<typename ColorType>
void SurfaceMeshRender<Scalar>::renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const std::vector< Color<ColorType> > &color)
{
    if(vertex_id.size()!= color.size())
    {
        std::cerr<<"warning: the size of vertex_id don't equal to color's, the vertex lacking of cunstom color will be rendered in white color !!"<<std::endl;
    }
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_POINT);             // enable polygon offset
    glPolygonOffset(-1.0,1.0);
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(static_cast<float>(1.5*point_size));

    glPushMatrix();
    if(this->transform_ != NULL)
    {
        openGLMultMatrix(this->transform_->transformMatrix());	
    }
 
    unsigned int num_vertex = vertex_id.size();
    glBegin(GL_POINTS);
    for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
    {
        if(vertex_idx<color.size())
            openGLColor3(color[vertex_idx]);
        else
            openGLColor3(Color<ColorType>::White());
        Vector<Scalar,3> position = this->mesh_->vertexPosition(vertex_id[vertex_idx]);
        openGLVertex(position);
    }
    glEnd();

    glPopMatrix();
    glPopAttrib();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::loadTextures()
{
    unsigned int num_material = this->mesh_->numMaterials();     // get the number of Material
    if(this->textures_.size() != num_material)
        this->textures_.resize(num_material);                  // resize the Array: textures_ ,which store the texture information from material, thus its size is equal to material size

    for(unsigned int material_idx=0; material_idx<num_material; material_idx++) // loop for ervery material
    {
        Material<Scalar> material_ref = this->mesh_->material(material_idx);    // get material reference


        if(material_ref.hasTexture())    // if have a texture
        {
            Image image;
            bool is_success = true;
            if(ImageIO::load(material_ref.textureFileName(), &image)== false) // load image data from file
            {
                std::cerr<<"error in loading image"<<std::endl;
                is_success = false;
            }
            std::pair<bool,unsigned int> texture;
            if( !is_success )        // if image_data is NULL, then set this material having no texture.
            {
                texture.first = false;
                this->textures_[material_idx] = texture;
                continue;
            }
            texture.first = true;
            glEnable(GL_TEXTURE_2D);
            glGenTextures(1, &(texture.second));                              // generate texture object
            glBindTexture(GL_TEXTURE_2D, texture.second);                     // bind the texture

            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);     // set paremeters GL_TEXTURE_WRAP_S
            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);     // set paremeters GL_TEXTURE_WRAP_T
            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); /// warning: we have to set the FILTER, otherwise the TEXTURE will "not" appear

            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_MODULATE);        /// warning
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image.rawData()); //generate texture 
            glDisable(GL_TEXTURE_2D);         

            this->textures_[material_idx] = texture; // add texture to Array textures_
        }
        else
        {
            std::pair<bool,unsigned int> texture;
            texture.first = false;
            this->textures_[material_idx] = texture;
        }
    }
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::releaseTextures()
{
    unsigned int num_material=this->textures_.size();     // get the number of Material
    for(unsigned int material_idx=0;material_idx<num_material;material_idx++)
    {
        if(this->textures_[material_idx].first==true)
        {
            this->textures_[material_idx].first=false;
            glDeleteTextures(1,&this->textures_[material_idx].second);
        }
    }
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::deleteDisplayLists()
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

//explicit instantitation
template class SurfaceMeshRender<float>;
template class SurfaceMeshRender<double>;

//for each color type: char,short,int,float,double,unsigned char,unsigned short,unsigned int
//explicit instantiate the render**WithColor Method
template void SurfaceMeshRender<float>::renderFaceWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void SurfaceMeshRender<float>::renderFaceWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void SurfaceMeshRender<float>::renderFaceWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void SurfaceMeshRender<float>::renderFaceWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void SurfaceMeshRender<float>::renderFaceWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void SurfaceMeshRender<float>::renderFaceWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void SurfaceMeshRender<float>::renderFaceWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void SurfaceMeshRender<float>::renderFaceWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void SurfaceMeshRender<double>::renderFaceWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);

template void SurfaceMeshRender<float>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void SurfaceMeshRender<float>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void SurfaceMeshRender<float>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void SurfaceMeshRender<float>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void SurfaceMeshRender<float>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void SurfaceMeshRender<float>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void SurfaceMeshRender<float>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void SurfaceMeshRender<float>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const Color<signed char>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const Color<short>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const Color<int>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const Color<float>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const Color<double>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const Color<unsigned char>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const Color<unsigned short>&);
template void SurfaceMeshRender<double>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const Color<unsigned int>&);

template void SurfaceMeshRender<float>::renderFaceWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char> >&);
template void SurfaceMeshRender<float>::renderFaceWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short> >&);
template void SurfaceMeshRender<float>::renderFaceWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int> >&);
template void SurfaceMeshRender<float>::renderFaceWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float> >&);
template void SurfaceMeshRender<float>::renderFaceWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double> >&);
template void SurfaceMeshRender<float>::renderFaceWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char> >&);
template void SurfaceMeshRender<float>::renderFaceWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short> >&);
template void SurfaceMeshRender<float>::renderFaceWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short> >&);
template void SurfaceMeshRender<double>::renderFaceWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int> >&);

template void SurfaceMeshRender<float>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char> >&);
template void SurfaceMeshRender<float>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short> >&);
template void SurfaceMeshRender<float>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int> >&);
template void SurfaceMeshRender<float>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float> >&);
template void SurfaceMeshRender<float>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double> >&);
template void SurfaceMeshRender<float>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char> >&);
template void SurfaceMeshRender<float>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short> >&);
template void SurfaceMeshRender<float>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<signed char>(const std::vector<unsigned int>&,const std::vector<Color<signed char> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<short>(const std::vector<unsigned int>&,const std::vector<Color<short> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<int>(const std::vector<unsigned int>&,const std::vector<Color<int> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<float>(const std::vector<unsigned int>&,const std::vector<Color<float> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<double>(const std::vector<unsigned int>&,const std::vector<Color<double> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<unsigned char>(const std::vector<unsigned int>&,const std::vector<Color<unsigned char> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<unsigned short>(const std::vector<unsigned int>&,const std::vector<Color<unsigned short> >&);
template void SurfaceMeshRender<double>::renderVertexWithColor<unsigned int>(const std::vector<unsigned int>&,const std::vector<Color<unsigned int> >&);

template void SurfaceMeshRender<float>::renderSolidWithCustomColor<signed char>(const std::vector<Color<signed char> >&);
template void SurfaceMeshRender<float>::renderSolidWithCustomColor<short>(const std::vector<Color<short> >&);
template void SurfaceMeshRender<float>::renderSolidWithCustomColor<int>(const std::vector<Color<int> >&);
template void SurfaceMeshRender<float>::renderSolidWithCustomColor<float>(const std::vector<Color<float> >&);
template void SurfaceMeshRender<float>::renderSolidWithCustomColor<double>(const std::vector<Color<double> >&);
template void SurfaceMeshRender<float>::renderSolidWithCustomColor<unsigned char>(const std::vector<Color<unsigned char> >&);
template void SurfaceMeshRender<float>::renderSolidWithCustomColor<unsigned short>(const std::vector<Color<unsigned short> >&);
template void SurfaceMeshRender<float>::renderSolidWithCustomColor<unsigned int>(const std::vector<Color<unsigned int> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<signed char>(const std::vector<Color<signed char> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<short>(const std::vector<Color<short> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<int>(const std::vector<Color<int> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<float>(const std::vector<Color<float> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<double>(const std::vector<Color<double> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<unsigned char>(const std::vector<Color<unsigned char> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<unsigned short>(const std::vector<Color<unsigned short> >&);
template void SurfaceMeshRender<double>::renderSolidWithCustomColor<unsigned int>(const std::vector<Color<unsigned int> >&);

} //end of namespace Physika
