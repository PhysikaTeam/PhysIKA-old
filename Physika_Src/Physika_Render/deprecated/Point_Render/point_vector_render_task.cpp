/*
 * @file point_render_task.h 
 * @Basic render task of point
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

#include <glm/glm.hpp>
#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "point_render_util.h"
#include "point_vector_render_shader_srcs.h"
#include "point_vector_render_task.h"

namespace Physika{


PointVectorRenderTask::PointVectorRenderTask(std::shared_ptr<PointRenderUtil> render_util)
    :render_util_(std::move(render_util))
{
    shader_.createFromCStyleString(point_vector_vertex_shader, point_vector_frag_shader, point_vector_geo_shader);
}


PointVectorRenderTask::~PointVectorRenderTask()
{

}

template <typename Scalar>
void PointVectorRenderTask::setUnifromPointVector(const Vector<Scalar, 3> & point_vector)
{
    vector_config_.setUniformPointVector(point_vector);
}

template <typename Scalar>
void PointVectorRenderTask::setPointVectors(const std::vector<Vector<Scalar, 3>> & point_vectors)
{
    if (point_vectors.size() != render_util_->pointNum())
        throw PhysikaException("error: point vector size not match point num!");

    vector_config_.setPointVectors(point_vectors);
}

void PointVectorRenderTask::enableUsePointVectorColor()
{
    use_point_vector_col_ = true;
}


void PointVectorRenderTask::disableUsePointVectorColor()
{
    use_point_vector_col_ = false;
}


bool PointVectorRenderTask::isUsePointVectorColor() const
{
    return use_point_vector_col_;
}


void PointVectorRenderTask::setColor(const Color4f & color)
{
    color_ = color;
}


const Color4f & PointVectorRenderTask::color() const
{
    return color_;
}


void PointVectorRenderTask::setScaleFactor(float scale_factor)
{
    scale_factor_ = scale_factor;
}


float PointVectorRenderTask::scaleFactor() const
{
    return scale_factor_;
}

void PointVectorRenderTask::setLineWidth(float line_width)
{
    line_width_ = line_width;
}

float PointVectorRenderTask::lineWidth() const
{
    return line_width_;
}
    
void PointVectorRenderTask::renderTaskImpl()
{
    openGLSetCurBindShaderBool("use_point_vector_col", use_point_vector_col_);
    openGLSetCurBindShaderCol3("col", color_);
    openGLSetCurBindShaderFloat("scale_factor", scale_factor_);

    this->configPointVectorToVertexVAO();

    glPushAttrib((GL_ALL_ATTRIB_BITS));
    glLineWidth((line_width_));

    render_util_->draw();

    glPopAttrib();
}

void PointVectorRenderTask::configPointVectorToVertexVAO()
{
    //bind point VAO
    render_util_->bindPointVAO();

    vector_config_.config(render_util_->pointNum());

    //unbind point VAO
    render_util_->unbindPointVAO();
}

//explicit instantiation
template void PointVectorRenderTask::setUnifromPointVector<float>(const Vector<float, 3> & point_vector);
template void PointVectorRenderTask::setUnifromPointVector<double>(const Vector<double, 3> & point_vector);
template void PointVectorRenderTask::setPointVectors<float>(const std::vector<Vector<float, 3>> & point_vectors);
template void PointVectorRenderTask::setPointVectors<double>(const std::vector<Vector<double, 3>> & point_vectors);
    
}//end of namespace Physika