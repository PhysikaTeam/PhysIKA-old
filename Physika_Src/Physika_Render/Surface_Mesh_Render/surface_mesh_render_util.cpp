/*
 * @file surface_mesh_render_util.cpp
 * @Brief class SurfaceMeshRenderUtil
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "surface_mesh_render_util.h"

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_Geometry/Boundary_Meshes/face_group.h"
#include "Physika_Geometry/Boundary_Meshes/material.h"
#include "Physika_IO/Image_IO/image_io.h"
#include "Physika_Core/Image/image.h"
#include "Physika_Core/Utilities/File_Utilities/file_path_utilities.h"

namespace Physika{

using BoundaryMeshInternal::Vertex;
using SurfaceMeshInternal::Face;
using SurfaceMeshInternal::FaceGroup;
using BoundaryMeshInternal::Material;

template <typename Scalar>
SurfaceMeshRenderUtil<Scalar>::SurfaceMeshRenderUtil(SurfaceMesh<Scalar> * mesh)
    :mesh_(mesh),
    point_render_util_(std::make_shared<PointRenderUtil>())
{
    this->initialize();
}

template <typename Scalar>
SurfaceMeshRenderUtil<Scalar>::~SurfaceMeshRenderUtil()
{
    this->destroy();
}

template <typename Scalar>
const SurfaceMesh<Scalar>* SurfaceMeshRenderUtil<Scalar>::mesh() const
{
    return mesh_;
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::setSurfaceMesh(SurfaceMesh<Scalar> *mesh)
{
    mesh_ = mesh;
    
    this->destroy();
    this->initialize();
}

template <typename Scalar>
std::shared_ptr<PointRenderUtil> SurfaceMeshRenderUtil<Scalar>::getInnerPointRenderUtil()
{
    return point_render_util_;
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::synchronize()
{
    this->destroy();
    this->initialize();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::drawBySolid()
{
    unsigned int group_num = this->mesh_->numGroups();
    if (group_num != group_render_utils_.size())
        throw PhysikaException("error: group num does not math group render util num!");

    for (unsigned int group_idx = 0; group_idx < group_num; ++group_idx)
    {
        FaceGroup<Scalar> & group_ref = this->mesh_->group(group_idx);
        unsigned int material_ID = group_ref.materialIndex();

        // get material property from mesh according to its materialIndex
        Vector<Scalar, 3> Ka = this->mesh_->material(material_ID).Ka();
        Vector<Scalar, 3> Kd = this->mesh_->material(material_ID).Kd();
        Vector<Scalar, 3> Ks = this->mesh_->material(material_ID).Ks();
        Scalar     shininess = this->mesh_->material(material_ID).shininess();
        Scalar         alpha = this->mesh_->material(material_ID).alpha();

        openGLSetCurBindShaderVec3("material.Ka", Ka);
        openGLSetCurBindShaderVec3("material.Kd", Kd);
        openGLSetCurBindShaderVec3("material.Ks", Ks);
        openGLSetCurBindShaderFloat("material.shininess", shininess);
        //openGLSetCurBindShaderFloat("material.alpha", alpha);

        //config texture
        if (this->group_textures_[material_ID].first == true)
        {
            glVerify(glActiveTexture(GL_TEXTURE0));
            glVerify(glBindTexture(GL_TEXTURE_2D, this->group_textures_[material_ID].second));

            openGLSetCurBindShaderBool("has_tex", true);
            openGLSetCurBindShaderInt("tex", 0);
        }

        //draw operation
        this->group_render_utils_[group_idx]->draw();

        //reset texture config
        if (this->group_textures_[material_ID].first == true)
        {
            glVerify(glActiveTexture(GL_TEXTURE0));
            glVerify(glBindTexture(GL_TEXTURE_2D, 0));

            openGLSetCurBindShaderBool("has_tex", false);
        }
    }
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::drawByPoint()
{
    point_render_util_->draw();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::bindPointVAO() const
{
    point_render_util_->bindPointVAO();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::unbindPointVAO() const
{
    point_render_util_->unbindPointVAO();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::bindGroupVAOAtIndex(unsigned int group_id)
{
    this->group_render_utils_[group_id]->bindTriangleVAO();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::unbindGroupVAOAtIndex(unsigned int group_id)
{
    this->group_render_utils_[group_id]->unbindTriangleVAO();
}

template <typename Scalar>
unsigned int SurfaceMeshRenderUtil<Scalar>::groupVAONum() const
{
    return this->group_render_utils_.size();
}

template <typename Scalar>
unsigned int SurfaceMeshRenderUtil<Scalar>::groupVertexNum(unsigned int group_id) const
{
    return this->group_render_utils_[group_id]->triangleNum() * 3;
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::initialize()
{
    this->initPointRenderUtil();
    this->initGroupRenderUtils();

    this->loadTextures();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::destroy()
{
    this->destroyGroupRenderUtils();

    this->releaseTextures();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::initPointRenderUtil()
{
    std::vector<Vector<Scalar, 3>> pos_vec;

    unsigned int vertex_num = this->mesh_->numVertices();
    for (unsigned int i = 0; i < vertex_num; ++i)
    {
        const Vector<Scalar, 3> & pos = this->mesh_->vertexPosition(i);
        pos_vec.push_back(pos);
    }
    
    point_render_util_->setPoints(pos_vec);
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::initGroupRenderUtils()
{
    unsigned int group_num = this->mesh_->numGroups();
    for (unsigned int group_idx = 0; group_idx < group_num; ++group_idx)
    {
        FaceGroup<Scalar> &group_ref = this->mesh_->group(group_idx);       // get group reference
        unsigned int face_num = group_ref.numFaces();                       // get face number

        std::vector<Vector<Scalar, 3>> pos_vec;
        std::vector<Vector<Scalar, 3>> normal_vec;
        std::vector<Vector<Scalar, 2>> tex_coord_vec;
        unsigned int group_vert_num = 0;

        for (unsigned int face_idx = 0; face_idx < face_num; ++face_idx)
        {
            Face<Scalar> &face_ref = group_ref.face(face_idx);          // get face reference
            unsigned int vertex_num = face_ref.numVertices();

            unsigned int triangle_num = vertex_num - 2;
            group_vert_num += 3 * triangle_num;

            for (unsigned int triangle_id = 0; triangle_id < triangle_num; ++triangle_id)
            {
                unsigned int triangle_vert_ids[3] = { 0, triangle_id + 1, triangle_id + 2 };
                for (unsigned vertex_idx : triangle_vert_ids)
                {
                    //pos
                    unsigned position_ID = face_ref.vertex(vertex_idx).positionIndex();   // get vertex positionIndex in "surface mesh"
                    const Vector<Scalar, 3> & pos = this->mesh_->vertexPosition(position_ID);
                    pos_vec.push_back(pos);

                    //normal
                    if (face_ref.vertex(vertex_idx).hasNormal())
                    {
                        unsigned int vertex_normal_ID = face_ref.vertex(vertex_idx).normalIndex();
                        const Vector<Scalar, 3> & normal = this->mesh_->vertexNormal(vertex_normal_ID);

                        normal_vec.push_back(normal);
                    }
                    else if (face_ref.hasFaceNormal())
                    {
                        const Vector<Scalar, 3> & face_normal = face_ref.faceNormal();
                        normal_vec.push_back(face_normal);
                    }

                    //tex coord
                    if (face_ref.vertex(vertex_idx).hasTexture())
                    {
                        unsigned int vertex_texture_ID = face_ref.vertex(vertex_idx).textureCoordinateIndex();
                        const Vector<Scalar, 2> & tex_coord = this->mesh_->vertexTextureCoordinate(vertex_texture_ID);

                        tex_coord_vec.push_back(tex_coord);
 
                    }
                }
            }

        }

        std::shared_ptr<TriangleRenderUtil> group_render_util = std::make_shared<TriangleRenderUtil>();

        group_render_util->setTriangles(pos_vec, false);

        if (normal_vec.size() > 0)
            group_render_util->setNormals(normal_vec);

        if (tex_coord_vec.size() > 0)
            group_render_util->setTexCoords(tex_coord_vec);

        this->group_render_utils_.push_back(std::move(group_render_util));
    }
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::destroyGroupRenderUtils()
{
    this->group_render_utils_.clear();
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::loadTextures()
{
    unsigned int num_material = this->mesh_->numMaterials();     // get the number of Material
    if(this->group_textures_.size() != num_material)
        this->group_textures_.resize(num_material);                    // resize the Array: group_textures_ ,which store the texture information from material, thus its size is equal to material size

    for(unsigned int material_idx=0; material_idx<num_material; material_idx++)  // loop for every material
    {
        Material<Scalar> &material_ref = this->mesh_->material(material_idx);    // get material reference

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
                this->group_textures_[material_idx] = texture;
                continue;
            }

            //if use png file, we vertically flip the image 
            if(FileUtilities::fileExtension(material_ref.textureFileName()) == ".png")
                image.flipVertically();
            
            texture.first = true;
            glGenTextures(1, &(texture.second));                              // generate texture object
            glBindTexture(GL_TEXTURE_2D, texture.second);                     // bind the texture

            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);     // set parameters GL_TEXTURE_WRAP_S
            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);     // set parameters GL_TEXTURE_WRAP_T
            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            openGLTexParameter(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image.rawData()); //generate texture 

            glBindTexture(GL_TEXTURE_2D, 0);

            this->group_textures_[material_idx] = texture; // add texture to Array group_textures_
            
        }
        else
        {
            std::pair<bool,unsigned int> texture;
            texture.first = false;
            this->group_textures_[material_idx] = texture;
        }
    }
}

template <typename Scalar>
void SurfaceMeshRenderUtil<Scalar>::releaseTextures()
{
    unsigned int num_material = this->group_textures_.size();     // get the number of Material
    for(unsigned int material_idx = 0;material_idx < num_material; material_idx++)
    {
        if(this->group_textures_[material_idx].first == true)
        {
            this->group_textures_[material_idx].first = false;
            glVerify(glDeleteTextures(1, &this->group_textures_[material_idx].second));
        }
    }
}


//explicit instantiation
template class SurfaceMeshRenderUtil<float>;
template class SurfaceMeshRenderUtil<double>;
    
}//end of namespace Physika