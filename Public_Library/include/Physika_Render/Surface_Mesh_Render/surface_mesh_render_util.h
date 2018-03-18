/*
 * @file surface_mesh_render_util.h 
 * @Brief class SurfaceMeshRenderUtil
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

#pragma once

#include <memory>
#include <vector>

#include "Physika_Render/Render_Util_Base/render_util_base.h"
#include "Physika_Render/Point_Render/point_render_util.h"
#include "Physika_Render/Triangle_Render/triangle_render_util.h"

namespace Physika{

template <typename Scalar> class SurfaceMesh;
template <typename Scalar, int Dim> class Transform;

class PointRenderUtil;
class TriangleRenderUtil;

template <typename Scalar>
class SurfaceMeshRenderUtil: public RenderUtilBase
{
public:

    SurfaceMeshRenderUtil() = default;
    SurfaceMeshRenderUtil(SurfaceMesh<Scalar> * mesh);
    ~SurfaceMeshRenderUtil();

    //disable copy
    SurfaceMeshRenderUtil(const SurfaceMeshRenderUtil & rhs) = delete;
    SurfaceMeshRenderUtil & operator = (const SurfaceMeshRenderUtil & rhs) = delete;

    //Get and Set
    const SurfaceMesh<Scalar>* mesh() const;
    void setSurfaceMesh(SurfaceMesh<Scalar> * mesh);

    std::shared_ptr<PointRenderUtil> getInnerPointRenderUtil();

    void synchronize() override;
    void drawBySolid() override;
    void drawByPoint() override;

    void bindPointVAO() const;
    void unbindPointVAO() const;

    void bindGroupVAOAtIndex(unsigned int group_id);
    void unbindGroupVAOAtIndex(unsigned int group_id);

    unsigned int groupVAONum() const;
    unsigned int groupVertexNum(unsigned int group_id) const;

private:
    void initialize();
    void destroy();

    void initPointRenderUtil();

    void initGroupRenderUtils();
    void destroyGroupRenderUtils();

    void loadTextures();
    void releaseTextures();

private:
    SurfaceMesh<Scalar> * mesh_ = nullptr;

    std::shared_ptr<PointRenderUtil> point_render_util_;
    std::vector<std::shared_ptr<TriangleRenderUtil>> group_render_utils_;

    std::vector<std::pair<bool, unsigned int> > group_textures_; //texture data, each element is a pair for one material of the mesh
                                                                 //the first entry is a flag indicating if there's texture for the material
                                                                 //the second entry is the OpenGL texture id

};
    
}//end of namespace Physika

