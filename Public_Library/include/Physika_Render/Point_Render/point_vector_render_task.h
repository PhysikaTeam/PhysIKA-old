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

#pragma once

#include <memory>
#include <vector>

#include "Physika_Render/Color/color.h"
#include "Physika_Render/Render_Task_Base/render_task_base.h"
#include "Physika_Render/Utilities/point_vector_attrib_config.h"


namespace Physika{

class PointRenderUtil;

class PointVectorRenderTask: public RenderTaskBase
{
public:
    explicit PointVectorRenderTask(std::shared_ptr<PointRenderUtil> render_util);
    ~PointVectorRenderTask();

    //disable copy
    PointVectorRenderTask(const PointVectorRenderTask &) = delete;
    PointVectorRenderTask & operator = (const PointVectorRenderTask &) = delete;

    template <typename Scalar>
    void setUnifromPointVector(const Vector<Scalar, 3> & point_vector);
    
    template <typename Scalar>
    void setPointVectors(const std::vector<Vector<Scalar, 3>> & point_vectors);

    void enableUsePointVectorColor();
    void disableUsePointVectorColor();
    bool isUsePointVectorColor() const;

    void setColor(const Color4f & color);
    const Color4f & color() const;

    void setScaleFactor(float scale_factor);
    float scaleFactor() const;

    void setLineWidth(float line_width);
    float lineWidth() const;

private:
    void renderTaskImpl() override;
    void configPointVectorToVertexVAO();

private:
    std::shared_ptr<PointRenderUtil> render_util_;

    PointVectorAttribConfig vector_config_;

    bool use_point_vector_col_ = false;
    Color4f color_ = Color4f::Red();

    float scale_factor_ = 1.0f;
    float line_width_ = 2.0f;
};
    
}
