/*
 * @file quad_mesh_point_render_task.h 
 * @Basic point render task of quad mesh
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

#include "Physika_Render/Point_Render/point_render_task.h"

namespace Physika{

template <typename Scalar>
class QuadMeshRenderUtil;

template <typename Scalar>
class QuadMeshPointRenderTask: public PointRenderTask
{
public:
    explicit QuadMeshPointRenderTask(std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util);
    ~QuadMeshPointRenderTask() = default;

    //disable copy
    QuadMeshPointRenderTask(const QuadMeshPointRenderTask &) = delete;
    QuadMeshPointRenderTask & operator = (const QuadMeshPointRenderTask &) = delete;

private:
    std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika