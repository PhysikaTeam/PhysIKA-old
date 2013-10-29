/*
 * @file surface_mesh_render.cpp 
 * @Basic render of surface mesh.
 * @author Sheng Yang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"

namespace Physika{
	
SurfaceMeshRender::SurfaceMeshRender()
{
    mesh_ = NULL;
}

SurfaceMeshRender::SurfaceMeshRender(SurfaceMesh<float>* mesh)
{
    mesh_ = mesh;
}

SurfaceMeshRender::~SurfaceMeshRender(void)
{

}

void SurfaceMeshRender::render()
{
    //To render the mesh in a screen based GLUI panel.
}

} //end of namespace Physika
