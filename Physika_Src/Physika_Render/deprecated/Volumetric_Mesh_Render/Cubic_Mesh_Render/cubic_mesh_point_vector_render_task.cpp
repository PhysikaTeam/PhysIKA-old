/*
 * @file cubic_mesh_point_render_task.cpp
 * @Basic point vector render task of cubic mesh
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


#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "cubic_mesh_render_util.h"
#include "cubic_mesh_point_vector_render_task.h"

namespace Physika{

template <typename Scalar>
CubicMeshPointVectorRenderTask<Scalar>::CubicMeshPointVectorRenderTask(std::shared_ptr<CubicMeshRenderUtil<Scalar>> render_util)
    :PointVectorRenderTask(render_util->getInnerPointRenderUtil()),
    render_util_(render_util)
{
    
}

template <typename Scalar>
void CubicMeshPointVectorRenderTask<Scalar>::setPointVectorAsBoundaryNormalVector()
{
    throw PhysikaException("error: not implemented yet");
}

//explicit instantiation 
template class CubicMeshPointVectorRenderTask<float>;
template class CubicMeshPointVectorRenderTask<double>;
    
}//end of namespace Physika