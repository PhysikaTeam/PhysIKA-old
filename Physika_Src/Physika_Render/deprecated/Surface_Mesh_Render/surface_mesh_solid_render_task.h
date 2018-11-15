/*
 * @file surface_mesh_solid_render_task.h 
 * @Basic solid render task base for surface mesh
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

#include <vector>
#include <memory>

#include "surface_mesh_custom_color_render_task_base.h"

namespace Physika {


template<typename Scalar>
class SurfaceMeshSolidRenderTask : public SurfaceMeshCustomColorRenderTaskBase<Scalar>
{
public:
    explicit SurfaceMeshSolidRenderTask(std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util);

    //disable copy
    SurfaceMeshSolidRenderTask(const SurfaceMeshSolidRenderTask &) = delete;
    SurfaceMeshSolidRenderTask & operator = (const SurfaceMeshSolidRenderTask &) = delete;

    //material
    void enableUseMaterial();
    void disableUseMaterial();
    bool isUseMaterial() const;

    //light
    void enableUseLight();
    void disableUseLight();
    bool isUseLight() const;

    //texture
    void enableUseTexture();
    void disableUseTexture();
    bool isUseTexture() const;

private:
    void customConfigs() override;

protected:
    std::shared_ptr<SurfaceMeshRenderUtil<Scalar>> render_util_;
    std::vector<VertexColorAttribConfig> color_configs_;

private:
    bool use_material_ = true;
    bool use_light_ = true;
    bool use_tex_ = true;
};

}//end of namespace Physika