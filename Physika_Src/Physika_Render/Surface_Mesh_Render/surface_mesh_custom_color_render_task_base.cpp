/*
 * @file surface_mesh_custom_color_render_task.cpp
 * @Basic custom_color render task of surface mesh
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

#include "Physika_Core/Utilities/glm_utilities.h"

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_Geometry/Boundary_Meshes/face_group.h"
#include "Physika_Geometry/Boundary_Meshes/material.h"

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "surface_mesh_render_util.h"
#include "surface_mesh_custom_color_render_task_base.h"

namespace Physika{

using BoundaryMeshInternal::Vertex;
using SurfaceMeshInternal::Face;
using SurfaceMeshInternal::FaceGroup;
using BoundaryMeshInternal::Material;

template <typename Scalar>
SurfaceMeshCustomColorRenderTaskBase<Scalar>::SurfaceMeshCustomColorRenderTaskBase(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util)
    :render_util_(std::move(render_util))
{
    for (unsigned int i = 0; i < render_util_->groupVAONum(); ++i)
        this->color_configs_.push_back({});

}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::enableUseCustomColor()
{
    use_custom_color_ = true;
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::disableUseCustomColor()
{
    use_custom_color_ = false;
}

template <typename Scalar>
bool SurfaceMeshCustomColorRenderTaskBase<Scalar>::isUseCustomColor() const
{
    return use_custom_color_;
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::setUniformColor(const Color4f & color)
{
    this->color_configs_.clear();
    for (unsigned int i = 0; i < render_util_->groupVAONum(); ++i)
    {
        this->color_configs_.push_back({});
        this->color_configs_[i].setUniformColor(color);
    }
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::setCustomVertexColors(const std::vector<Color4f> & vert_cols)
{
    this->destroyColorAttribConfigs();
    this->initColorAttribConfigsByVertexColors(vert_cols);
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::setCustomFaceColors(const std::vector<Color4f> & face_cols)
{
    this->destroyColorAttribConfigs();
    this->initColorAttribConfigsByFaceColors(face_cols);
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::setCustomGroupColors(const std::vector<Color4f> & group_cols)
{
    this->destroyColorAttribConfigs();
    this->initColorAttribConfigsByGroupColors(group_cols);
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::renderTaskImpl()
{
    this->configColorToGroupVAOs();
    openGLSetCurBindShaderBool("use_custom_color", use_custom_color_);

    glVerify(glPushAttrib(GL_ALL_ATTRIB_BITS));

    this->customConfigs();
    render_util_->drawBySolid();

    glVerify(glPopAttrib());
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::initColorAttribConfigsByVertexColors(const std::vector<Color4f> & vert_cols)
{
    const SurfaceMesh<Scalar> * mesh = render_util_->mesh();
    if (mesh->numVertices() != vert_cols.size())
        throw PhysikaException("error: size of vert colors not match vertex num in mesh!");

    unsigned int group_num = mesh->numGroups();
    for (unsigned int group_idx = 0; group_idx < group_num; ++group_idx)
    {
        const FaceGroup<Scalar> & group_ref = mesh->group(group_idx);
        unsigned int face_num = group_ref.numFaces();

        std::vector<glm::vec3> col_vec;

        for (unsigned int face_idx = 0; face_idx < face_num; ++face_idx)
        {
            const Face<Scalar> & face_ref = group_ref.face(face_idx);          // get face reference
            unsigned int vertex_num = face_ref.numVertices();

            unsigned int triangle_num = vertex_num - 2;

            for (unsigned int triangle_id = 0; triangle_id < triangle_num; ++triangle_id)
            {
                unsigned int triangle_vert_ids[3] = { 0, triangle_id + 1, triangle_id + 2 };
                for (unsigned int vertex_idx : triangle_vert_ids)
                {
                    unsigned int position_ID = face_ref.vertex(vertex_idx).positionIndex();
                    const Color4f & col = vert_cols[position_ID];
                    col_vec.push_back(convertCol3(col));
                }
            }
        }

        VertexColorAttribConfig group_color_config;
        group_color_config.setColors(col_vec);

        this->color_configs_.push_back(std::move(group_color_config));
    }
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::initColorAttribConfigsByFaceColors(const std::vector<Color4f> & face_cols)
{
    const SurfaceMesh<Scalar> * mesh = render_util_->mesh();
    if (mesh->numFaces() != face_cols.size())
        throw PhysikaException("error: size of face colors not match face num in mesh!");

    unsigned int cur_global_face_id = 0;
    unsigned int group_num = mesh->numGroups();

    for (unsigned int group_idx = 0; group_idx < group_num; ++group_idx)
    {
        const FaceGroup<Scalar> & group_ref = mesh->group(group_idx);
        unsigned int face_num = group_ref.numFaces();

        std::vector<glm::vec3> col_vec;

        for (unsigned int face_idx = 0; face_idx < face_num; ++face_idx, ++cur_global_face_id)
        {
            const Face<Scalar> & face_ref = group_ref.face(face_idx);
            unsigned int vertex_num = face_ref.numVertices();

            unsigned int triangle_num = vertex_num - 2;
            const Color4f & col = face_cols[cur_global_face_id];
            glm::vec3 glm_col = convertCol3(col);

            col_vec.insert(col_vec.end(), 3 * triangle_num, glm_col);
        }

        VertexColorAttribConfig group_color_config;
        group_color_config.setColors(col_vec);

        this->color_configs_.push_back(std::move(group_color_config));
    }

}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::initColorAttribConfigsByGroupColors(const std::vector<Color4f> & group_cols)
{
    const SurfaceMesh<Scalar> * mesh = render_util_->mesh();
    if (mesh->numGroups() != group_cols.size())
        throw PhysikaException("error: size of face colors not match group num in mesh!");

    unsigned int group_num = mesh->numGroups();
    for (unsigned int i = 0; i < group_num; ++i)
    {
        this->color_configs_.push_back({});
        this->color_configs_[i].setUniformColor(group_cols[i]);
    }
}

template <typename Scalar>
void  SurfaceMeshCustomColorRenderTaskBase<Scalar>::destroyColorAttribConfigs()
{
    this->color_configs_.clear();
}

template <typename Scalar>
void SurfaceMeshCustomColorRenderTaskBase<Scalar>::configColorToGroupVAOs()
{
    if (this->color_configs_.size() != render_util_->groupVAONum())
        throw PhysikaException("error: vertex color config size not match group VAO size, please call setCustomVertexColors first");

    for (unsigned int i = 0; i < this->color_configs_.size(); ++i)
    {
        //bind VAO
        render_util_->bindGroupVAOAtIndex(i);

        this->color_configs_[i].config(render_util_->groupVertexNum(i));

        //unbind VAO
        render_util_->unbindGroupVAOAtIndex(i);
    }
}

//explicit instantiation
template class SurfaceMeshCustomColorRenderTaskBase<float>;
template class SurfaceMeshCustomColorRenderTaskBase<double>;

}//end of namespace Physika