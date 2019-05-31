/*
 * @file point_render_task.h 
 * @Basic render task of point
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

#include <Physika_Core/Array/Array.h>
#include "CudaVBOMapper.h"
#include "ShaderProgram.h"

namespace Physika{

class PointRender
{
public:
    explicit PointRender();
    ~PointRender();

    //disable copy
	PointRender(const PointRender &) = delete;
	PointRender & operator = (const PointRender &) = delete;

	void resize(unsigned int num);

	void setVertexArray(DeviceArray<float3>& pos);
	void setVertexArray(HostArray<float3>& pos);

    void setPointSize(float point_size);
    float pointSize() const;

	void setColor(glm::vec3 color);
	void setColor(DeviceArray<glm::vec3> color);

    void setPointScaleForPointSprite(float point_scale);
    float pointScaleForPointSprite() const;

    void enableUsePointSprite();
    void disableUsePointSprite();
    bool isUsePointSprite() const;

	void display();

private:
    bool use_point_sprite_ = true;

    float point_size_ = 1.0f;
    float point_scale_ = 5.0f; //for point sprite

	CudaVBOMapper<glm::vec3> m_vertVBO;
	CudaVBOMapper<glm::vec3> m_normVBO;
	CudaVBOMapper<glm::vec3> m_vertexColor;

	ShaderProgram m_glsl;
//	GLSLProgram*  nvGLSL;

};
    
}
