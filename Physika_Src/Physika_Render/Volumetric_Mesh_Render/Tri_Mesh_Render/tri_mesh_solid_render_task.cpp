/*
 * @file tri_mesh_solid_render_task.cpp
 * @Basic solid render task of tri mesh
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

#include "tri_mesh_render_util.h"
#include "tri_mesh_solid_render_task.h"

namespace Physika{

template <typename Scalar>
TriMeshSolidRenderTask<Scalar>::TriMeshSolidRenderTask(std::shared_ptr<TriMeshRenderUtil<Scalar>> render_util)
    :TriangleSolidRenderTask(render_util->getInnerTriangleRenderUtil()),
    render_util_(render_util)
{
    
}

//explicit instantiation 
template class TriMeshSolidRenderTask<float>;
template class TriMeshSolidRenderTask<double>;
    
}//end of namespace Physika