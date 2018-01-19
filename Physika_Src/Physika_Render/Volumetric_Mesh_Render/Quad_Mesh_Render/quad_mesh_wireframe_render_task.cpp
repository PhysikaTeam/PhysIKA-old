/*
 * @file quad_mesh_wireframe_render_task.cpp
 * @Basic wireframe render task of quad mesh
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
#include "quad_mesh_wireframe_render_task.h"

namespace Physika{

template <typename Scalar>
QuadMeshWireframeRenderTask<Scalar>::QuadMeshWireframeRenderTask(std::shared_ptr<QuadMeshRenderUtil<Scalar>> render_util)
    :QuadWireframeRenderTask(render_util->getInnerQuadRenderUtil()),
    render_util_(render_util)
{
    
}

//explicit instantiation 
template class QuadMeshWireframeRenderTask<float>;
template class QuadMeshWireframeRenderTask<double>;
    
}//end of namespace Physika