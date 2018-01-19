/*
 * @file surface_mesh_wireframe_render_task.cpp
 * @Basic wireframe render task of surface mesh
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


#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/Triangle_Render/triangle_wireframe_render_shader_srcs.h"

#include "surface_mesh_render_util.h"
#include "surface_mesh_wireframe_render_task.h"

namespace Physika{

template <typename Scalar>
SurfaceMeshWireframeRenderTask<Scalar>::SurfaceMeshWireframeRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util)
    :SurfaceMeshCustomColorRenderTaskBase(std::move(render_util))
{
    shader_.createFromCStyleString(triangle_wireframe_render_vertex_shader,triangle_wireframe_render_frag_shader);
}

template <typename Scalar>
void SurfaceMeshWireframeRenderTask<Scalar>::setLineWidth(float line_width)
{
    line_width_ = line_width;
}

template <typename Scalar>
float SurfaceMeshWireframeRenderTask<Scalar>::lineWidth() const
{
    return line_width_;
}

template <typename Scalar>
void SurfaceMeshWireframeRenderTask<Scalar>::customConfigs()
{
    glLineWidth(line_width_);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

//explicit instantiation
template class SurfaceMeshWireframeRenderTask<float>;
template class SurfaceMeshWireframeRenderTask<double>;

}//end of namespace Physika