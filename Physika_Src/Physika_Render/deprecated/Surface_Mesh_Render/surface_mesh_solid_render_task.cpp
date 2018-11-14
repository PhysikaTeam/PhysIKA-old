/*
 * @file surface_mesh_solid_render_task.cpp
 * @Basic solid render task base for surface mesh
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
#include "Physika_Render/Triangle_Render/triangle_solid_render_shader_srcs.h"

#include "surface_mesh_render_util.h"
#include "surface_mesh_solid_render_task.h"

namespace Physika {

template<typename Scalar>
SurfaceMeshSolidRenderTask<Scalar>::SurfaceMeshSolidRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util)
    :SurfaceMeshCustomColorRenderTaskBase<Scalar>(std::move(render_util))
{
    this->shader_.createFromCStyleString(triangle_solid_render_vertex_shader, triangle_solid_render_frag_shader);
}


template <typename Scalar>
void SurfaceMeshSolidRenderTask<Scalar>::enableUseMaterial()
{
    use_material_ = true;
}

template <typename Scalar>
void SurfaceMeshSolidRenderTask<Scalar>::disableUseMaterial()
{
    use_material_ = false;
}

template <typename Scalar>
bool SurfaceMeshSolidRenderTask<Scalar>::isUseMaterial() const
{
    return use_material_;
}

template <typename Scalar>
void SurfaceMeshSolidRenderTask<Scalar>::enableUseLight()
{
    use_light_ = true;
}

template <typename Scalar>
void SurfaceMeshSolidRenderTask<Scalar>::disableUseLight()
{
    use_light_ = false;
}

template <typename Scalar>
bool SurfaceMeshSolidRenderTask<Scalar>::isUseLight() const
{
    return use_light_;
}

template <typename Scalar>
void SurfaceMeshSolidRenderTask<Scalar>::enableUseTexture()
{
    use_tex_ = true;
}

template <typename Scalar>
void SurfaceMeshSolidRenderTask<Scalar>::disableUseTexture()
{
    use_tex_ = false;
}

template <typename Scalar>
bool SurfaceMeshSolidRenderTask<Scalar>::isUseTexture() const
{
    return use_tex_;
}

template<typename Scalar>
void SurfaceMeshSolidRenderTask<Scalar>::customConfigs()
{
    openGLSetCurBindShaderBool("use_material", use_material_);
    openGLSetCurBindShaderBool("use_light", use_light_);
    openGLSetCurBindShaderBool("use_tex", use_tex_);
}

//explicit instantiation
template class SurfaceMeshSolidRenderTask<float>;
template class SurfaceMeshSolidRenderTask<double>;

}//end of namespace Physika