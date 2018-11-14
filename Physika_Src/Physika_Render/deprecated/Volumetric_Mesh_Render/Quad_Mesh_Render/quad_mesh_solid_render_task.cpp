/*
 * @file quad_mesh_solid_render_task.cpp
 * @Basic solid render task of quad mesh
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

#include "quad_mesh_render_util.h"
#include "quad_mesh_solid_render_task.h"

namespace Physika{

template <typename Scalar>
QuadMeshSolidRenderTask<Scalar>::QuadMeshSolidRenderTask(std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util)
    :QuadSolidRenderTask(render_util->getInnerQuadRenderUtil()),
    render_util_(render_util)
{
    
}

//explicit instantiation 
template class QuadMeshSolidRenderTask<float>;
template class QuadMeshSolidRenderTask<double>;
    
}//end of namespace Physika