/*
 * @file surface_mesh_wireframe_render_task.h 
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

#pragma once

#include <memory>

#include "surface_mesh_custom_color_render_task_base.h"

namespace Physika{

template <typename Scalar>
class SurfaceMeshRenderUtil;

template <typename Scalar>
class SurfaceMeshWireframeRenderTask : public SurfaceMeshCustomColorRenderTaskBase<Scalar>
{
public:
    SurfaceMeshWireframeRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util);

    void setLineWidth(float line_width);
    float lineWidth() const;

private:
    void customConfigs() override;

private:

    float line_width_ = 2.0f;
};
    
}//end of namespace Physika