/*
 * @file surface_mesh_render.h 
 * @Basic render of mesh.
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

#ifndef PHYSIKA_RENDER_SURFACE_MESH_RENDER_SURFACE_MESH_RENDER_H_
#define PHYSIKA_RENDER_SURFACE_MESH_RENDER_SURFACE_MESH_RENDER_H_

#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"

namespace Physika{

class SurfaceMeshRender: public RenderBase
{
public:
    /* Constructions */
    SurfaceMeshRender();
    SurfaceMeshRender(SurfaceMesh<float>* mesh);
    ~SurfaceMeshRender(void);

    /* Get and Set */
    inline SurfaceMesh<float>* mesh() { return mesh_; }

    /* Render */
    virtual void render();
protected:
    
    /* Render data*/
    SurfaceMesh<float> *mesh_;
};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_SURFACE_MESH_RENDER_SURFACE_MESH_RENDER_H_
