/*
 * @file surface_mesh_point_render_task.cpp
 * @Basic render task of surface mesh vertex
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

#include "surface_mesh_render_util.h"
#include "surface_mesh_point_render_task.h"

namespace Physika{

template <typename Scalar>
SurfaceMeshPointRenderTask<Scalar>::SurfaceMeshPointRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util)
    :PointRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(std::move(render_util))
{

}

//explicit instantiation
template class SurfaceMeshPointRenderTask<float>;
template class SurfaceMeshPointRenderTask<double>;

    
}//end of namespace Physika