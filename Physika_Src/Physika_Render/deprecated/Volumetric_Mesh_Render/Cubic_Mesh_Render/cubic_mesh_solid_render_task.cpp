/*
 * @file cubic_mesh_solid_render_task.cpp
 * @Basic solid render task of cubic mesh
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

#include "cubic_mesh_render_util.h"
#include "cubic_mesh_solid_render_task.h"

namespace Physika{

template <typename Scalar>
CubicMeshSolidRenderTask<Scalar>::CubicMeshSolidRenderTask(std::shared_ptr<CubicMeshRenderUtil<Scalar>> render_util)
    :CubeSolidRenderTask(render_util->getInnerCubeRenderUtil()),
    render_util_(render_util)
{
    
}

//explicit instantiation 
template class CubicMeshSolidRenderTask<float>;
template class CubicMeshSolidRenderTask<double>;
    
}//end of namespace Physika