/*
 * @file surface_mesh_point_render_task.h 
 * @Basic render task of line
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

#include <glm/glm.hpp>
#include <GL/glew.h>
//#include "Physika_Render/OpenGL_Primitives/glew_utilities.h"

#include "PointRender.h"

namespace Physika{

const char * vertexSource = R"STR(
#version 330 compatibility
layout(location = 0) in vec3 vert_pos;
layout(location = 3) in vec3 vert_col;

out vec3 frag_vert_col;

uniform mat4 proj_trans;
uniform mat4 view_trans;
uniform mat4 model_trans;

uniform bool use_point_sprite;
uniform float point_size;
uniform float point_scale;

void main()
{
	frag_vert_col = vert_col;

	gl_Position = gl_ModelViewProjectionMatrix * vec4(vert_pos, 1.0);
	if (use_point_sprite)
	{
		gl_PointSize = point_scale * point_size / gl_Position.w;
	}
}
)STR";

const char * fragmentSource = R"STR(
#version 330 compatibility
in vec3 frag_vert_col;
out vec4 frag_color;

uniform bool use_point_sprite;

void main()
{
	if (use_point_sprite)
	{
		vec3 normal;
		//normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
		normal.xy = gl_PointCoord.xy * 2.0 - 1.0;
		float mag = dot(normal.xy, normal.xy);
		if (mag > 1.0) discard;
		normal.z = sqrt(1.0 - mag);

		vec3 light_dir = vec3(0, 1, 0);
		float diffuse_factor = max(dot(light_dir, normal), 0);
		vec3 diffuse = diffuse_factor * frag_vert_col;

		diffuse = vec3(normal.z) * frag_vert_col;
		frag_color = vec4(diffuse, 1.0);
	}
	else
	{
		frag_color = vec4(frag_vert_col, 1.0);
	}
}
)STR";

// vertex shader
const char *vertexShader1 = R"STR(
uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform float densityScale;
uniform float densityOffset;
void main()
{
	// calculate window-space point size
	vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
	float dist = length(posEye);
	gl_PointSize = pointRadius * (pointScale / dist);

	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	gl_FrontColor = gl_Color;
}
);

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader1 = STRINGIFY(
	void main()
{
	const vec3 lightDir = vec3(0.577, 0.577, 0.577);

	// calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(N.xy, N.xy);

	if (mag > 1.0) discard;   // kill pixels outside circle

	N.z = sqrt(1.0 - mag);

	// calculate lighting
	float diffuse = max(0.0, dot(lightDir, N));

	gl_FragColor = gl_Color * diffuse;
}
)STR";

PointRender::PointRender()
{
	m_glsl.createFromCStyleString(vertexSource, fragmentSource);
//	nvGLSL = new GLSLProgram(vertexShader1, spherePixelShader1);
}

PointRender::~PointRender()
{
}

void PointRender::resize(unsigned int num)
{
	m_vertVBO.resize(num);
 	m_vertexColor.resize(num);
// 	m_normVBO.resize(num);
}


void PointRender::setVertexArray(DeviceArray<float3>& pos)
{
	cudaMemcpy(m_vertVBO.cudaMap(), pos.getDataPtr(), sizeof(float3) * pos.size(), cudaMemcpyDeviceToDevice);
	m_vertVBO.cudaUnmap();
}

void PointRender::setVertexArray(HostArray<float3>& pos)
{
	cudaMemcpy(m_vertVBO.cudaMap(), pos.getDataPtr(), sizeof(float3) * pos.size(), cudaMemcpyDeviceToHost);
	m_vertVBO.cudaUnmap();
}

void PointRender::setColorArray(DeviceArray<float3>& color)
{
	cudaMemcpy(m_vertexColor.cudaMap(), color.getDataPtr(), sizeof(float3) * color.size(), cudaMemcpyDeviceToDevice);
	m_vertexColor.cudaUnmap();
}

void PointRender::setColorArray(HostArray<float3>& color)
{
	cudaMemcpy(m_vertexColor.cudaMap(), color.getDataPtr(), sizeof(float3) * color.size(), cudaMemcpyDeviceToHost);
	m_vertexColor.cudaUnmap();
}

void PointRender::setPointSize(float point_size)
{
    point_size_ = point_size;
}

float PointRender::pointSize() const
{
    return point_size_;
}

void PointRender::setColor(glm::vec3 color)
{
	glm::vec3* colors = new glm::vec3[m_vertexColor.getSize()];
	for (size_t i = 0; i < m_vertexColor.getSize(); i++)
	{
		colors[i] = color;
	}

	cudaMemcpy(m_vertexColor.cudaMap(), colors, sizeof(glm::vec3) * m_vertexColor.getSize(), cudaMemcpyHostToHost);
	m_vertexColor.cudaUnmap();
	delete[] colors;
}

void PointRender::setPointScaleForPointSprite(float point_scale)
{
    point_scale_ = point_scale;
}

float PointRender::pointScaleForPointSprite() const
{
    return point_scale_;
}

void PointRender::enableUsePointSprite()
{
    use_point_sprite_ = true;
}

void PointRender::disableUsePointSprite()
{
    use_point_sprite_ = false;
}

bool PointRender::isUsePointSprite() const
{
    return use_point_sprite_;
}

void PointRender::display()
{
	m_glsl.enable();

	Matrix4f idenity = Matrix4f::identityMatrix();
	m_glsl.setMat4("proj_trans", idenity);
	m_glsl.setMat4("view_trans", idenity);
	m_glsl.setMat4("model_trans", idenity);

	Vector3f vec = Vector3f(0.0f);
	m_glsl.setVec3("view_pos", vec);

	glPushAttrib(GL_ALL_ATTRIB_BITS);

	if (use_point_sprite_)
	{
		glEnable(GL_POINT_SPRITE);
		glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glEnable(GL_PROGRAM_POINT_SIZE);

		m_glsl.setFloat("point_size", point_size_);
		m_glsl.setFloat("point_scale", point_scale_);
		m_glsl.setBool("use_point_sprite", true);
	}
	else
	{
		glPointSize(point_size_);
	}

	glEnableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexColor.getVBO());
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertVBO.getVBO());
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_POINTS, 0, m_vertVBO.getSize());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);


	m_glsl.disable();
}


}//end of namespace Physika