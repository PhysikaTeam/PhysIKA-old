/*
 * @file surface_mesh_render.cpp 
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

#include <cstddef>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"

namespace Physika{

//init render mode flags
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_solid_ = 1<<0;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_wireframe_ = 1<<1;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_vertices_ = 1<<2;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_flat_or_smooth_ = 1<<3;
template <typename Scalar> const unsigned int SurfaceMeshRender<Scalar>::render_texture_ = 1<<5;
	
template <typename Scalar>
SurfaceMeshRender<Scalar>::SurfaceMeshRender()
    :mesh_(NULL)
{
    initRenderMode();
}

template <typename Scalar>
SurfaceMeshRender<Scalar>::SurfaceMeshRender(SurfaceMesh<Scalar>* mesh)
    :mesh_(mesh)
{
    initRenderMode();
    loadTextures();
}

template <typename Scalar>
SurfaceMeshRender<Scalar>::~SurfaceMeshRender()
{
    releaseTextures();
}

template <typename Scalar>
const SurfaceMesh<Scalar>* SurfaceMeshRender<Scalar>::mesh() const
{
    return mesh_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::setSurfaceMesh(SurfaceMesh<Scalar> *mesh)
{
    mesh_ = mesh;
    //after updating the mesh, the textures needed to be update correspondently
    releaseTextures();
    loadTextures();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableRenderSolid()
{
    render_mode_ |= render_solid_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableRenderSolid()
{
    render_mode_ &= ~render_solid_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableRenderVertices()
{
    render_mode_ |= render_vertices_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableRenderVertices()
{
    render_mode_ &= ~render_vertices_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableRenderWireframe()
{
    render_mode_ |= render_wireframe_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableRenderWireframe()
{
    render_mode_ &= ~render_wireframe_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableFlatShading()
{
    render_mode_ &= ~render_flat_or_smooth_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableSmoothShading()
{
    render_mode_ |= render_flat_or_smooth_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::enableTexture()
{
    render_mode_ |= render_texture_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::disableTexture()
{
    render_mode_ &= ~render_texture_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::render()
{
    PHYSIKA_MESSAGE_ASSERT(mesh_!=NULL,"No mesh is passed to the MeshRender!");
    if(render_mode_ & render_solid_)
	renderSolid();
    if(render_mode_ & render_wireframe_)
	renderWireframe();
    if(render_mode_ & render_vertices_)
	renderVertices();
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::initRenderMode()
{
    //default render mode: solid, smooth shading, texture
    render_mode_ = 0;
    render_mode_ |= render_solid_;
    render_mode_ |= render_flat_or_smooth_;
    render_mode_ |= render_texture_;
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderVertices()
{
//TO DO: render vertices
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderWireframe()
{
//TO DO: render wireframe
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::renderSolid()
{
//TO DO: render solid mode
//WARNING: 
//        1. choose different vertex normals according to render_flat_or_smooth_
//        2. render texture according to render_texture_
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::loadTextures()
{
//TO DO: implementation
}

template <typename Scalar>
void SurfaceMeshRender<Scalar>::releaseTextures()
{
//TO DO: implementation
}

//explicit instantitation
template class SurfaceMeshRender<float>;
template class SurfaceMeshRender<double>;

} //end of namespace Physika















