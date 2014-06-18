/*
 * @file volumetric_mesh_render.cpp 
 * @Brief render of volumetric mesh.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_render.h"

namespace Physika{

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::VolumetricMeshRender()
    :mesh_(NULL)
{
}

template <typename Scalar, int Dim>
VolumetricMeshRender<Scalar,Dim>::~VolumetricMeshRender()
{
}

}  //end of namespace Physika
