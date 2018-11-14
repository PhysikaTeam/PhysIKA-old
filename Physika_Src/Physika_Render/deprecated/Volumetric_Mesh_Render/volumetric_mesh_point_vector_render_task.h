/*
 * @file volumetric_mesh_point_vector_render_task.h 
 * @Basic point_vector render task of volumetric mesh
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

#include "Physika_Render/Point_Render/point_vector_render_task.h"

namespace Physika{

template <typename Scalar, int Dim>
class VolumetricMeshRenderUtil;

template <typename Scalar, int Dim>
class VolumetricMeshPointVectorRenderTask: public PointVectorRenderTask
{
public:
    explicit VolumetricMeshPointVectorRenderTask(std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util_);
    ~VolumetricMeshPointVectorRenderTask() = default;

    //disable copy
    VolumetricMeshPointVectorRenderTask(const VolumetricMeshPointVectorRenderTask &) = delete;
    VolumetricMeshPointVectorRenderTask & operator = (const VolumetricMeshPointVectorRenderTask &) = delete;

private:
    std::shared_ptr<VolumetricMeshRenderUtil<Scalar, Dim>> render_util_;

};
    
}//end of namespace Physika