/*
 * @file surface_mesh_normal_render_task.h 
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

#pragma once

#include <memory>
#include "Physika_Render/Render_Task_Base/render_task_base.h"

namespace Physika{

template <typename Scalar>
class SurfaceMeshRenderUtil;

template <typename Scalar>
class SurfaceMeshNormalRenderTask: public RenderTaskBase
{
public:
    SurfaceMeshNormalRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util);

    void enableMapToColorSpace();
    void disableMapToColorSpace();
    bool isMapToColorSpace() const;

private:
    void renderTaskImpl() override;

private:
    std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util_;
    bool map_to_color_space_ = false;
};
    
}//end of namespace Physika