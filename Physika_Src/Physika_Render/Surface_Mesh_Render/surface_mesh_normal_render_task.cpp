/*
 * @file surface_mesh_normal_render_task.cpp
 * @Basic render task of surface mesh for visual normal vector 
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

#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render_util.h"
#include "surface_mesh_normal_render_shader_srcs.h"

#include "surface_mesh_normal_render_task.h"

namespace Physika{

template <typename Scalar>
SurfaceMeshNormalRenderTask<Scalar>::SurfaceMeshNormalRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util)
    :render_util_(std::move(render_util))
{
    shader_.createFromCStyleString(surface_mesh_normal_render_vertex_shader, surface_mesh_normal_render_frag_shader);
}

template <typename Scalar>
void SurfaceMeshNormalRenderTask<Scalar>::enableMapToColorSpace()
{
    map_to_color_space_ = true;
}

template <typename Scalar>
void SurfaceMeshNormalRenderTask<Scalar>::disableMapToColorSpace()
{
    map_to_color_space_ = false;
}

template <typename Scalar>
bool SurfaceMeshNormalRenderTask<Scalar>::isMapToColorSpace() const
{
    return map_to_color_space_;
}


template <typename Scalar>
void SurfaceMeshNormalRenderTask<Scalar>::renderTaskImpl()
{
    openGLSetCurBindShaderBool("map_to_color_space", map_to_color_space_);
    render_util_->drawBySolid();

}

//explicit instantiation
template class SurfaceMeshNormalRenderTask<float>;
template class SurfaceMeshNormalRenderTask<double>;
    
}