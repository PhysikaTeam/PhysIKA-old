/*
 * @file volumetric_mesh_point_render_task.h 
 * @Basic point render task of volumetric mesh
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

template <typename Scalar, int Dim>
class VolumetricMeshRenderUtil;

template <typename Scalar, int Dim>
class VolumetricMeshPointRenderTask: public PointRenderTask
{
public:
    explicit VolumetricMeshPointRenderTask(std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util_);
    ~VolumetricMeshPointRenderTask() = default;

    //disable copy
    VolumetricMeshPointRenderTask(const VolumetricMeshPointRenderTask &) = delete;
    VolumetricMeshPointRenderTask & operator = (const VolumetricMeshPointRenderTask &) = delete;

private:
    std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util_;

};
    
}//end of namespace Physika