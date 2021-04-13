/*
 * @file triangle_solid_render_task.h 
 * @Basic render task of triangle
 * @author Wei Chen, Xiaowei He
 * 
 * This file is part of PhysIKA, a versatile physics simulation library.
 * Copyright (C) 2013- PhysIKA Group.
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

namespace PhysIKA{

class TriangleRender
{
public:
    explicit TriangleRender();
    ~TriangleRender() = default;

    //disable copy
    TriangleRender(const TriangleRender &) = delete;
    TriangleRender & operator = (const TriangleRender &) = delete;

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

private:
    bool use_custom_color_ = true;
	int m_lineWidth = 2;

	bool m_bShowWireframe = false;
	bool m_bEnableLighting = false;
	bool m_bEnableDoubleShading = true;

	ShaderProgram m_solidShader;
	ShaderProgram m_wireframeShader;

	CudaVBOMapper<glm::vec3> m_vertVBO;
	CudaVBOMapper<glm::vec3> m_normVBO;
	CudaVBOMapper<glm::vec3> m_colorVBO;
};
    
}