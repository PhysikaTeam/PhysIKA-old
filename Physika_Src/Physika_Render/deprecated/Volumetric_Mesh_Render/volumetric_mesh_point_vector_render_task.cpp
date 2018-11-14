/*
 * @file volumetric_mesh_point_vector_render_task.cpp 
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

#include "volumetric_mesh_render_util.h"
#include "volumetric_mesh_point_vector_render_task.h"

namespace Physika{

template <typename Scalar, int Dim>
VolumetricMeshPointVectorRenderTask<Scalar, Dim>::VolumetricMeshPointVectorRenderTask(std::shared_ptr<VolumetricMeshRenderUtil<Scalar,Dim>> render_util)
    :PointVectorRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(render_util)
{
    
}

//explicit instantiation
template class VolumetricMeshPointVectorRenderTask<float, 2>;
template class VolumetricMeshPointVectorRenderTask<float, 3>;
template class VolumetricMeshPointVectorRenderTask<double, 2>;
template class VolumetricMeshPointVectorRenderTask<double, 3>;
    
}//end of namespace Physika