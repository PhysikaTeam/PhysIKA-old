/*
 * @file surface_mesh_render.h 
 * @Brief render of surface mesh.
 * @author Fei Zhu, Wei Chen
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
#include "Physika_Core/Arrays/array.h"
#include "Physika_Render/Render_Base/render_base.h"

namespace Physika{

template <typename Scalar> class SurfaceMesh;
template <typename Scalar> class Color;

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

    //whenever the mesh is modified, synchronize() must be called to update the render
    void synchronize();   

    //Render with choosen render mode
    virtual void render();

    //The following four functions is used for rendering specific faces/vertices with custom color.
    // 1: when you employ " vector< Color<float> > color" to sepcify your cunstom color, in the case of the color size smaller than
    //    face_id/vertex_id 's , the face/vertex lacking of color will be rendered in default(white) color.
    // 2: when you render specific vertices, the GL_POINT_SIZE will be "1.5" times of the orignal one so that you can 
    //    distinguish the new specific vertices from the old one.
    template<typename ColorType>
    void renderFaceWithColor(const std::vector<unsigned int> &face_id, const Color<ColorType> &color);
    template<typename ColorType>
    void renderFaceWithColor(const std::vector<unsigned int> &face_id, const std::vector< Color<ColorType> > &color);

    template<typename ColorType>
    void renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const Color<ColorType> &color);
    template<typename ColorType>
    void renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const std::vector< Color<ColorType> > &color);

	template<typename ColorType>
	void renderSolidWithCustomColor(const std::vector< Color<ColorType> > & color);

protected:
    void initRenderMode();
    void renderVertices();
    void renderWireframe();
    void renderSolid();

    void loadTextures();
    void releaseTextures();
    void deleteDisplayLists();
protected:
    //render mode
    unsigned int render_mode_;
    //data
    SurfaceMesh<Scalar> *mesh_;
    //texture data, each element is a pair for one material of the mesh
    //the first entry is a flag indicating if there's texture for the material
    //the second entry is the OpenGL texture id
    Array<std::pair<bool,unsigned int> > textures_;

    //displaylist ids
    unsigned int vertex_display_list_id_;   
    unsigned int wire_display_list_id_;     
    unsigned int solid_display_list_id_;
	unsigned int face_with_color_display_list_id_;
	unsigned int face_with_color_vector_display_list_id_;
	unsigned int vertex_with_color_display_list_id_;
	unsigned int vertex_with_color_vector_display_list_id_;
	unsigned int solid_with_custom_color_vector_display_list_id_;

    //predefined render modes
    static const unsigned int render_solid_;
    static const unsigned int render_wireframe_;
    static const unsigned int render_vertices_;
    static const unsigned int render_flat_or_smooth_;  //1 bit: smooth, 0 bit: flat
    static const unsigned int render_texture_;
};

} //end of namespace Physika

#endif //PHYSIKA_RENDER_SURFACE_MESH_RENDER_SURFACE_MESH_RENDER_H_
