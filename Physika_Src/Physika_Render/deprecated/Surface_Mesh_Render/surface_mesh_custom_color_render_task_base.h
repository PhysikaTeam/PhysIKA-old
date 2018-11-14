/*
 * @file surface_mesh_custom_color_render_task.h 
 * @Basic custom_color render task of surface mesh
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
#include "Physika_Render/Render_Task_Base/render_task_base.h"
#include "Physika_Render/Utilities/vertex_color_attrib_config.h"

namespace Physika{

template <typename Scalar>
class SurfaceMeshRenderUtil;

template <typename Scalar>
class SurfaceMeshCustomColorRenderTaskBase : public RenderTaskBase
{
public:
    SurfaceMeshCustomColorRenderTaskBase(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util);

    //disable copy
    SurfaceMeshCustomColorRenderTaskBase(const SurfaceMeshCustomColorRenderTaskBase &) = delete;
    SurfaceMeshCustomColorRenderTaskBase & operator = (const SurfaceMeshCustomColorRenderTaskBase &) = delete;

    void enableUseCustomColor();
    void disableUseCustomColor();
    bool isUseCustomColor() const;

    void setUniformColor(const Color4f & color);
    void setCustomVertexColors(const std::vector<Color4f> & vert_cols);
    void setCustomFaceColors(const std::vector<Color4f> & face_cols);
    void setCustomGroupColors(const std::vector<Color4f> & group_cols);


private:
    void renderTaskImpl() final;
    virtual void customConfigs() = 0; //hook to custom configs

    void initColorAttribConfigsByVertexColors(const std::vector<Color4f> & vert_cols);
    void initColorAttribConfigsByFaceColors(const std::vector<Color4f> & face_cols);
    void initColorAttribConfigsByGroupColors(const std::vector<Color4f> & group_cols);

    void destroyColorAttribConfigs();
    void configColorToGroupVAOs();

private:
    std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util_;
    std::vector<VertexColorAttribConfig> color_configs_;
    bool use_custom_color_ = false;
};
    
}//end of namespace Physika