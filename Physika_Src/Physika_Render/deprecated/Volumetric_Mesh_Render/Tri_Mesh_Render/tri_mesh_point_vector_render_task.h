/*
 * @file tri_mesh_point_vector_render_task.h 
 * @Basic point vector render task of tri mesh
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
class TriMeshRenderUtil;

template <typename Scalar>
class TriMeshPointVectorRenderTask: public PointVectorRenderTask
{
public:
    explicit TriMeshPointVectorRenderTask(std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util);
    ~TriMeshPointVectorRenderTask() = default;

    //disable copy
    TriMeshPointVectorRenderTask(const TriMeshPointVectorRenderTask &) = delete;
    TriMeshPointVectorRenderTask & operator = (const TriMeshPointVectorRenderTask &) = delete;

    void setPointVectorAsBoundaryNormalVector();

private:
    std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util_;
};

}//end of namespace Physika