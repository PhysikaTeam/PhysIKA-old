/*
 * @file volumetric_mesh_render.h 
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

#ifndef PHYSIKA_RENDER_VOLUMETRIC_MESH_RENDER_VOLUMETRIC_MESH_RENDER_H_
#define PHYSIKA_RENDER_VOLUMETRIC_MESH_RENDER_VOLUMETRIC_MESH_RENDER_H_

#include "Physika_Render/Render_Base/render_base.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;

template <typename Scalar, int Dim>
class VolumetricMeshRender: public RenderBase
{
public:
    //constructions
    VolumetricMeshRender();
    ~VolumetricMeshRender();
protected:
    VolumetricMesh<Scalar,Dim> *mesh_;
};

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_VOLUMETRIC_MESH_RENDER_VOLUMETRIC_MESH_RENDER_H_
