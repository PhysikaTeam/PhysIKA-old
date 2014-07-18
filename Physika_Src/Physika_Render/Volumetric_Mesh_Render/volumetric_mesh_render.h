/*
 * @file volumetric_mesh_render.h 
 * @Brief render of volumetric mesh.
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

#ifndef PHYSIKA_RENDER_VOLUMETRIC_MESH_RENDER_VOLUMETRIC_MESH_RENDER_H_
#define PHYSIKA_RENDER_VOLUMETRIC_MESH_RENDER_VOLUMETRIC_MESH_RENDER_H_

#include "Physika_Render/Render_Base/render_base.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;
template <typename Scalar> class Color;
template <typename Scalar, int Dim> class Transform;

template <typename Scalar, int Dim>
class VolumetricMeshRender: public RenderBase
{
public:
    //constructions
    VolumetricMeshRender();
    VolumetricMeshRender(VolumetricMesh<Scalar, Dim> *mesh);
    VolumetricMeshRender(VolumetricMesh<Scalar, Dim> *mesh, Transform<Scalar, Dim> *transform);
    //destruction
    ~VolumetricMeshRender();

    //Get and Set
    const VolumetricMesh<Scalar, Dim>* mesh()const;
    void setVolumetricMesh(VolumetricMesh<Scalar, Dim>* mesh);
    void setVolumetricMesh(VolumetricMesh<Scalar, Dim>* mesh, Transform<Scalar, Dim>* transform);

    const Transform<Scalar, Dim>* transform()const;
    void setTransform(Transform<Scalar, Dim>* transform);

    //set render mode
    void enableRenderSolid();
    void disableRenderSolid();
    void enableRenderVertices();
    void disableRenderVertices();
    void enableRenderWireframe();
    void disableRenderWireframe();

    //whenever the mesh is modified, synchronize() must be called to update the render
    void synchronize(); 

    //Render with choosen render mode
    virtual void render();

    // note: renderSolidWithAlpha is a function to render volumetricMesh in solid mode, with a alpha value to represent the transparency.
    //       a value of 0.0 denotes that the mesh is completely transparent.
    //       naturally ,a increasing alpha value means the mesh has a higher opacity. 
    //       while an alpha value of 1.0 is not meaning an absolute opacity strictly, which also depends on other states such as background color you are specifying.
    //       default is 0.5 as shown in the follows
    void renderSolidWithAlpha(float alpha=0.5);

    //The following four functions is used for rendering specific elements/vertices with custom color.
    // 1: when you employ " vector< Color<float> > color" to sepcify your cunstom color, in the case of the color size smaller than
    //    element_id/vertex_id 's , the face/vertex lacking of color will be rendered in default(white) color.
    // 2: when you render specific vertices, the GL_POINT_SIZE will be "1.5" times of the orignal one so that you can 
    //    distinguish the new specific vertices from the old one.
    template<typename ColorType>
    void renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const Color<ColorType> &color);
    template<typename ColorType>
    void renderVertexWithColor(const std::vector<unsigned int> &vertex_id, const std::vector< Color<ColorType> > &color);

    template<typename ColorType>
    void renderElementWithColor(const std::vector<unsigned int> &element_id, const Color<ColorType> &color);
    template<typename ColorType>
    void renderElementWithColor(const std::vector<unsigned int> &element_id, const std::vector< Color<ColorType> > &color);

    template<typename ColorType>
    void renderSolidWithCustomColor(const std::vector< Color<ColorType> > & color);

protected:
    void initRenderMode();
    void renderVertices();
    void renderWireframe();

    void deleteDisplayLists();

    void drawTriOrQuad(unsigned int ele_idx);
    void drawTet(unsigned int ele_idx);
    void drawCubic(unsigned int ele_idx);
protected:
    unsigned int render_mode_;
    VolumetricMesh<Scalar,Dim> *mesh_;
    Transform<Scalar, Dim> *transform_;


    //displaylist ids
    unsigned int vertex_display_list_id_;   
    unsigned int wire_display_list_id_;     
    unsigned int solid_display_list_id_;
    unsigned int solid_with_custom_color_vector_display_list_id_;

    //predefined render modes
    static const unsigned int render_solid_;
    static const unsigned int render_wireframe_;
    static const unsigned int render_vertices_;
};

}  //end of namespace Physika

#endif //PHYSIKA_RENDER_VOLUMETRIC_MESH_RENDER_VOLUMETRIC_MESH_RENDER_H_
