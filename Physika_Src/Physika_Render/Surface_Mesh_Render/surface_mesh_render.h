/*
 * @file surface_mesh_render.h 
 * @Basic render of surface mesh.
 * @author Sheng Yang, Fei Zhu
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

#include <utility>
#include "Physika_Core/Array/array.h"
#include "Physika_Render/Render_Base/render_base.h"
using std::pair;

namespace Physika{

template <typename Scalar> class SurfaceMesh;

template <typename Scalar>
class SurfaceMeshRender: public RenderBase
{
public:
    //Constructions
    SurfaceMeshRender();
    //the parameter is not const because renderer may call method of mesh to modify its normals
    SurfaceMeshRender(SurfaceMesh<Scalar>* mesh);
    ~SurfaceMeshRender();

    //Get and Set
    const SurfaceMesh<Scalar>* mesh() const;
    void setSurfaceMesh(SurfaceMesh<Scalar>* mesh);
    //set render mode
    void enableRenderSolid();
    void disableRenderSolid();
    void enableRenderVertices();
    void disableRenderVertices();
    void enableRenderWireframe();
    void disableRenderWireframe();
    void enableFlatShading();
    void enableSmoothShading();
    void enableTexture();
    void disableTexture();

    //Render
    virtual void render();
protected:
    void initRenderMode();
    void renderVertices();
    void renderWireframe();
    void renderSolid();
    void loadTextures();
    void releaseTextures();
protected:
    //render mode
    unsigned int render_mode_;
    //data
    SurfaceMesh<Scalar> *mesh_;
    //texture data, each element is a pair for one material of the mesh
    //the first entry is a flag indicating if there's texture for the material
    //the second entry is the OpenGL texture id
    Array<pair<bool,unsigned int> > textures_;

    //predefined render modes
    static const unsigned int render_solid_;
    static const unsigned int render_wireframe_;
    static const unsigned int render_vertices_;
    static const unsigned int render_flat_or_smooth_;  //1 bit: smooth, 0 bit: flat
    static const unsigned int render_texture_;
};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_SURFACE_MESH_RENDER_SURFACE_MESH_RENDER_H_













