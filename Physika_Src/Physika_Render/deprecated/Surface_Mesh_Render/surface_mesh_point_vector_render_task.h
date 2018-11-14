/*
 * @file surface_mesh_point_vector_render_task.h 
 * @Basic render task of surface mesh vertex with custom vector
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

template <typename Scalar>
class SurfaceMeshRenderUtil;

template <typename Scalar>
class SurfaceMeshPointVectorRenderTask: public PointVectorRenderTask
{
public:
    SurfaceMeshPointVectorRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util);
    ~SurfaceMeshPointVectorRenderTask() = default;

    //disable copy
    SurfaceMeshPointVectorRenderTask(const SurfaceMeshPointVectorRenderTask &) = delete;
    SurfaceMeshPointVectorRenderTask & operator = (const SurfaceMeshPointVectorRenderTask &) = delete;

    void setPointVectorsAsNormalVector();

private:
    std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util_;
};
    
}//end of namespace Physika