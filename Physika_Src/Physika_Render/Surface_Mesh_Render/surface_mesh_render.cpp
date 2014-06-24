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


namespace Physika{

//init render mode flags
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_solid_ = 1<<0;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_wireframe_ = 1<<1;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_vertices_ = 1<<2;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_flat_or_smooth_ = 1<<3;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_texture_ = 1<<5;
    
template <typename Scalar>
SurfaceMeshRender<Scalar>::SurfaceMeshRender()
    :mesh_(NULL),solid_display_list_id_(0),
     wire_display_list_id_(0),vertex_display_list_id_(0)
{
    initRenderMode();
}

template <typename Scalar>
SurfaceMeshRender<Scalar>::SurfaceMeshRender(SurfaceMesh<Scalar>* mesh)
    :mesh_(mesh),solid_display_list_id_(0),
     wire_display_list_id_(0),vertex_display_list_id_(0)
{
    initRenderMode();
    loadTextures();
}

template <typename Scalar>
SurfaceMeshRender<Scalar>::~SurfaceMeshRender()
{
    releaseTextures();
    deleteDisplayLists();
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
    glPopAttrib();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderWireframe()
{ 
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);        // set openGL polygon mode for wire mode
    glDisable(GL_LIGHTING);

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
    glPopAttrib();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderSolid()
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT);
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // set polygon mode FILL for SOLID MODE
    glDisable(GL_COLOR_MATERIAL);              /// warning: we have to disable GL_COLOR_MATERIAL, otherwise the material propertity won't appear!!!
    glEnable(GL_LIGHTING);                     
    glEnable(GL_LIGHT0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  //glClear()
    
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

    glPopAttrib();
}

template <typename Scalar> template<typename glScalar>
void SurfaceMeshRender<Scalar>::renderFaceWithColor(std::vector<unsigned int> face_id, Color<glScalar> color)
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);     // set polygon mode FILL for SOLID MODE
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_FILL);              // enable polygon offset
    openGLColor3(color);
    glPolygonOffset(-1.0,1.0);                      // set polygon offset (factor, unit)
    
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
}

template <typename Scalar> template<typename glScalar>
void SurfaceMeshRender<Scalar>::renderFaceWithColor(std::vector<unsigned int> face_id, std::vector< Color<glScalar> > color)
{
    if(face_id.size()!= color.size())
    {
        std::cout<<"warning: the size of face_id don't equal to color's, the face lacking of cunstom color will be rendered in black color !"<<std::endl;
    }
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);     // set polygon mode FILL for SOLID MODE
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_FILL);              // enable polygon offset
    glPolygonOffset(-1.0,1.0);                      // set polygon offset (factor, unit)
    
    unsigned int num_face = face_id.size();     
    for(unsigned int face_idx=0; face_idx<num_face; face_idx++)
    {
        const Face<Scalar>& face_ref = this->mesh_->face(face_id[face_idx]);      //get the reference of face with face_id: face_idx
        unsigned int num_vertex = face_ref.numVertices();
        if(face_idx<color.size())
            openGLColor3(color[face_idx]);
        else
            openGLColor3(Color<glScalar>());
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

}

template <typename Scalar> template<typename glScalar>
void SurfaceMeshRender<Scalar>::renderVertexWithColor(std::vector<unsigned int> vertex_id, Color<glScalar> color)
{
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_POINT);             // enable polygon offset
    glPolygonOffset(-1.0,1.0); 
    openGLColor3(color);
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(1.5*point_size);
    std::cout<<"point_size:"<<point_size<<std::endl;
    unsigned int num_vertex = vertex_id.size();
    glBegin(GL_POINTS);
    for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
    {
        Vector<Scalar,3> position = this->mesh_->vertexPosition(vertex_id[vertex_idx]); // get the position of vertex which is stored in "surface mesh"
        openGLVertex(position);
    }
    glEnd();
    glPopAttrib();
}

template <typename Scalar> template<typename glScalar>
void SurfaceMeshRender<Scalar>::renderVertexWithColor(std::vector<unsigned int> vertex_id, std::vector< Color<glScalar> > color)
{
    if(vertex_id.size()!= color.size())
    {
        std::cout<<"warning: the size of vertex_id don't equal to color's, the vertex lacking of cunstom color will be rendered in black color !!"<<std::endl;
    }
    glPushAttrib(GL_LIGHTING_BIT|GL_POLYGON_BIT|GL_ENABLE_BIT|GL_TEXTURE_BIT|GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT|GL_POINT_BIT);
    glDisable(GL_LIGHTING);                        /// turn light off, otherwise the color may not appear
    glEnable(GL_POLYGON_OFFSET_POINT);             // enable polygon offset
    glPolygonOffset(-1.0,1.0);
    float point_size;
    glGetFloatv(GL_POINT_SIZE,&point_size);
    glPointSize(1.5*point_size);
    
    unsigned int num_vertex = vertex_id.size();
    glBegin(GL_POINTS);
    for(unsigned int vertex_idx=0; vertex_idx<num_vertex; vertex_idx++)
    {
        if(vertex_idx<color.size())
            openGLColor3(color[vertex_idx]);
        else
            openGLColor3(Color<glScalar>());
        Vector<Scalar,3> position = this->mesh_->vertexPosition(vertex_id[vertex_idx]);
        openGLVertex(position);
    }
    glEnd();
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
            int width,height;
            unsigned char * image_data = ImageIO::load(material_ref.textureFileName(),width,height); // load image data from file
            std::pair<bool,unsigned int> texture;
            if(image_data==NULL)        // if image_data is NULL, then set this material having no texture.
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

            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,  GL_MODULATE);       /// warning
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data); //generate texture 
            glDisable(GL_TEXTURE_2D);         

            this->textures_[material_idx] = texture; // add texture to Array textures_
            delete(image_data);                      //free image data
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

    this->solid_display_list_id_ = 0;
    this->wire_display_list_id_ = 0;
    this->vertex_display_list_id_ = 0;	
}

//explicit instantitation
template class SurfaceMeshRender<float>;
template class SurfaceMeshRender<double>;


template void SurfaceMeshRender<float>::
    renderFaceWithColor<float>(std::vector<unsigned int> face_id,  Color<float> color);
template void SurfaceMeshRender<float>::
    renderFaceWithColor<double>(std::vector<unsigned int> face_id, Color<double> color);
template void SurfaceMeshRender<double>::
    renderFaceWithColor<float>(std::vector<unsigned int> face_id, Color<float> color);
template void SurfaceMeshRender<double>::
    renderFaceWithColor<double>(std::vector<unsigned int> face_id, Color<double> color);

template void SurfaceMeshRender<float>::
    renderVertexWithColor<float>(std::vector<unsigned int> vertex_id, Color<float> color);
template void SurfaceMeshRender<float>::
    renderVertexWithColor<double>(std::vector<unsigned int> vertex_id, Color<double> color);
template void SurfaceMeshRender<double>::
    renderVertexWithColor<float>(std::vector<unsigned int> vertex_id, Color<float> color);
template void SurfaceMeshRender<double>::
    renderVertexWithColor<double>(std::vector<unsigned int> vertex_id, Color<double> color);


template void SurfaceMeshRender<float>::
    renderFaceWithColor<float>(std::vector<unsigned int> face_id, std::vector< Color<float> > color);
template void SurfaceMeshRender<float>::
    renderFaceWithColor<double>(std::vector<unsigned int> face_id, std::vector< Color<double> > color);
template void SurfaceMeshRender<double>::
    renderFaceWithColor<float>(std::vector<unsigned int> face_id, std::vector< Color<float> > color);
template void SurfaceMeshRender<double>::
    renderFaceWithColor<double>(std::vector<unsigned int> face_id, std::vector< Color<double> > color);

template void SurfaceMeshRender<float>::
    renderVertexWithColor<float>(std::vector<unsigned int> vertex_id, std::vector< Color<float> > color);
template void SurfaceMeshRender<float>::
    renderVertexWithColor<double>(std::vector<unsigned int> vertex_id, std::vector< Color<double> > color);
template void SurfaceMeshRender<double>::
    renderVertexWithColor<float>(std::vector<unsigned int> vertex_id, std::vector< Color<float> > color);
template void SurfaceMeshRender<double>::
    renderVertexWithColor<double>(std::vector<unsigned int> vertex_id, std::vector< Color<double> > color);

} //end of namespace Physika
