/*
 * @file triangle_solid_render_task.h 
 * @Basic render task of triangle
 * @author Wei Chen, Xiaowei He
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
#include "CudaVBOMapper.h"
#include <Core/Array/Array.h>
#include "ShaderProgram.h"
#include "Core/Quaternion/quaternion.h"

namespace PhysIKA {

class RigidTriangleRender
{
public:
    explicit RigidTriangleRender();
    ~RigidTriangleRender() = default;

    //disable copy
    RigidTriangleRender(const RigidTriangleRender&) = delete;
    RigidTriangleRender& operator=(const RigidTriangleRender&) = delete;

    void setVertexArray(HostArray<float3>& vertArray);
    void setVertexArray(DeviceArray<float3>& vertArray);

    void setNormalArray(HostArray<float3>& normArray);
    void setNormalArray(DeviceArray<float3>& normArray);

    void setColorArray(HostArray<float3>& colorArray);
    void setColorArray(DeviceArray<float3>& colorArray);

    void enableDoubleShading();
    void disableDoubleShading();

    void enableUseCustomColor();
    void disableUseCustomColor();
    bool isUseCustomColor() const;

    void display();

    void resize(unsigned int triNum);

    void setRotation(const Quaternion<float>& q)
    {
        m_rotation = q;
    }
    void setTranslatioin(const Vector3f& t)
    {
        m_translation = t;
    }
    void setScale(const Vector3f& s)
    {
        m_scale = s;
    }

private:
    bool use_custom_color_ = true;
    int  m_lineWidth       = 2;

    bool m_bShowWireframe       = false;
    bool m_bEnableLighting      = false;
    bool m_bEnableDoubleShading = true;

    ShaderProgram m_solidShader;
    ShaderProgram m_wireframeShader;

    CudaVBOMapper<glm::vec3> m_vertVBO;
    CudaVBOMapper<glm::vec3> m_normVBO;
    CudaVBOMapper<glm::vec3> m_colorVBO;

    Quaternion<float> m_rotation;
    Vector3f          m_scale;
    Vector3f          m_translation;
};

}  // namespace PhysIKA