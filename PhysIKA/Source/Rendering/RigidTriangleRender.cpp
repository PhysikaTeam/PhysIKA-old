/*
 * @file triangle_solid_render_task.h 
 * @Basic solid render task of triangle
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

#include <GL/glew.h>
#include "RigidTriangleRender.h"

//#include "Core/Matrix/general_matrix.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace PhysIKA {

	static const char * triangle_rigid_render_vertex_shader = R"STR(
#version 330

layout(location = 0) in vec3 VertexPosition;
layout(location = 1) in vec3 VertexNormal;
layout(location = 2) in vec3 VertexColor;

out vec3 LightIntensity;
out vec3 outVertexColor;

struct LightInfo
{
	vec4 Position; //ligth position in eye coords
	vec3 La; //ambient light intensity
	vec3 Ld; //diffuse light intensity
	vec3 Ls; // specular light intensity
};
uniform LightInfo Light;

struct MaterialInfo
{
	vec3 Ka; // ambient reflectivity
	vec3 Kd; // diffuse reflectivity
	vec3 Ks; // specular reflectivity
	float Shininess; // specular shininess factor
};
uniform MaterialInfo Material;

uniform bool bDoubleShading;
uniform mat4 ModelViewMatrix;
uniform mat3 NormalMatrix;
uniform mat4 ProjectionMatrix;
uniform mat4 MVP;
uniform mat4 ModelMat;


void main()
{
	vec3 tnorm = normalize(NormalMatrix * VertexNormal);
	vec4 eyeCoords = ModelViewMatrix * vec4(VertexPosition, 1.0);
	vec3 s = normalize(vec3(Light.Position - eyeCoords));
	vec3 v = normalize(-eyeCoords.xyz);
	vec3 r = reflect(-s, tnorm);
	float sDotN = dot(s, tnorm);	
	if(bDoubleShading)
		sDotN = abs(sDotN);
	else
		sDotN = max(0.0, dot(s, tnorm));
	vec3 ambient = Light.La * Material.Ka;
	vec3 diffuse = Light.Ld * Material.Ka * sDotN;
	vec3 spec = vec3(0.0);
	if (sDotN > 0.0)
	{
		spec = Light.Ls * Material.Ks *
			pow(max(0.0, dot(r, v)), Material.Shininess);
	}
	LightIntensity = ambient + diffuse + spec;
	
	gl_Position = MVP * ( ModelMat * vec4(VertexPosition, 1.0));
	outVertexColor = VertexColor;
})STR";

static const char * triangle_rigid_render_frag_shader = R"STR(
	#version 330 

	in vec3 LightIntensity;
	in vec3 outVertexColor;

	layout(location = 0) out vec4 FragColor;

	void main()
	{
		FragColor = vec4(LightIntensity*outVertexColor, 1.0);
	}
	)STR";


static const char * triangle_rigid_wireframe_render_vertex_shader = R"STR(
	#version 330 compatibility
	layout(location = 0) in vec3 vert_pos;
	layout(location = 3) in vec3 vert_col;

	out vec3 frag_vert_col;

	void main()
	{
		frag_vert_col = vert_col;
		gl_Position = gl_ModelViewProjectionMatrix * vec4(vert_pos, 1.0);
	}
	)STR";

static const char * triangle_wireframe_render_frag_shader = R"STR(
	#version 330 compatibility
	in vec3 frag_vert_col;
	out vec4 frag_color;

	void main()
	{
		frag_color = vec4(frag_vert_col, 1.0);
	}
	)STR";

RigidTriangleRender::RigidTriangleRender()
{
	m_solidShader.createFromCStyleString(triangle_rigid_render_vertex_shader, triangle_rigid_render_frag_shader);
	m_wireframeShader.createFromCStyleString(triangle_rigid_wireframe_render_vertex_shader, triangle_wireframe_render_frag_shader);
}

void RigidTriangleRender::setVertexArray(HostArray<float3>& vertArray)
{
	cudaMemcpy(m_vertVBO.cudaMap(), vertArray.getDataPtr(), sizeof(float3) * m_vertVBO.getSize(), cudaMemcpyHostToDevice);
	m_vertVBO.cudaUnmap();
}

void RigidTriangleRender::setVertexArray(DeviceArray<float3>& vertArray)
{
	cudaMemcpy(m_vertVBO.cudaMap(), vertArray.getDataPtr(), sizeof(float3) * m_vertVBO.getSize(), cudaMemcpyDeviceToDevice);
	m_vertVBO.cudaUnmap();
}

void RigidTriangleRender::setNormalArray(HostArray<float3>& normArray)
{
	cudaMemcpy(m_normVBO.cudaMap(), normArray.getDataPtr(), sizeof(float3) * m_normVBO.getSize(), cudaMemcpyHostToDevice);
	m_normVBO.cudaUnmap();
}

void RigidTriangleRender::setNormalArray(DeviceArray<float3>& normArray)
{
	cudaMemcpy(m_normVBO.cudaMap(), normArray.getDataPtr(), sizeof(float3) * m_normVBO.getSize(), cudaMemcpyDeviceToDevice);
	m_normVBO.cudaUnmap();
}

void RigidTriangleRender::setColorArray(HostArray<float3>& colorArray)
{
	cudaMemcpy(m_colorVBO.cudaMap(), colorArray.getDataPtr(), sizeof(float3) * m_colorVBO.getSize(), cudaMemcpyHostToDevice);
	m_colorVBO.cudaUnmap();
}

void RigidTriangleRender::setColorArray(DeviceArray<float3>& colorArray)
{
	cudaMemcpy(m_colorVBO.cudaMap(), colorArray.getDataPtr(), sizeof(float3) * m_colorVBO.getSize(), cudaMemcpyDeviceToDevice);
	m_colorVBO.cudaUnmap();
}

void RigidTriangleRender::enableDoubleShading()
{
	m_bEnableDoubleShading = true;
}

void RigidTriangleRender::disableDoubleShading()
{
	m_bEnableDoubleShading = false;
}

void RigidTriangleRender::enableUseCustomColor()
{
    use_custom_color_ = true;
}

void RigidTriangleRender::disableUseCustomColor()
{
    use_custom_color_ = false;
}

bool RigidTriangleRender::isUseCustomColor() const
{
    return use_custom_color_;
}


void RigidTriangleRender::display()
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	if (m_bShowWireframe)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glLineWidth(m_lineWidth);
		m_wireframeShader.enable();
	}
	else
	{
		m_solidShader.enable();


		//glm::mat4 model_rotate=glm::make_mat4(m_rotation.get4x4Matrix().getDataPtr());// = m_rotation.getGlm4x4Matrix();
		//glm::vec3 model_translate(m_translation[0], m_translation[1], m_translation[2]);
		//glm::mat4 model_mat = glm::translate(model_rotate, glm::vec3(m_translation[0], m_translation[1], m_translation[2]));
		float ran;
		Vector3f axis;
		m_rotation.toRotationAxis(ran, axis);
		glm::mat4 model_mat(1.0);
		model_mat = glm::translate(model_mat, glm::vec3(m_translation[0], m_translation[1], m_translation[2]));
		model_mat = glm::rotate(model_mat, ran, glm::vec3(axis[0], axis[1], axis[2]));

		glm::mat4 mvMat;
		glm::mat3 normMat;
		glm::mat4 projMat;
		glm::mat4 viewMat;
		glGetFloatv(GL_MODELVIEW_MATRIX, &mvMat[0][0]);
		glGetFloatv(GL_PROJECTION_MATRIX, &projMat[0][0]);
		normMat = glm::mat3(glm::vec3(mvMat[0]), glm::vec3(mvMat[1]), glm::vec3(mvMat[2]));
		m_solidShader.setMat4("ModelViewMatrix", mvMat);
		m_solidShader.setMat3("NormalMatrix", normMat);
		m_solidShader.setMat4("ProjectionMatrix", projMat);
		m_solidShader.setMat4("MVP", projMat * mvMat);
		m_solidShader.setBool("bDoubleShading", m_bEnableDoubleShading);


		//model_rotate = glm::identity<glm::mat4>();
		//model_rotate = glm::make_mat4(Quaternion<float>(Vector3f(0,0,1), 1.57).get4x4Matrix().getDataPtr());
		//model_translate[0] = 1; model_translate[1] = 0; model_translate[2] = 0;

		//m_solidShader.setMat4("ModelRotate", model_rotate);
		//m_solidShader.setVec3("ModelTranslate", model_translate);
		m_solidShader.setMat4("ModelMat", model_mat);

		glm::vec4  worldLight = glm::vec4(-5.0f, 5.0f, 2.0f, 1.0f);
		m_solidShader.setVec3("Material.Kd", 0.9f, 0.5f, 0.3f);
		m_solidShader.setVec3("Light.Ld", 1.0f, 1.0f, 1.0f);
		m_solidShader.setVec4("Light.Position", mvMat*worldLight);
		m_solidShader.setVec3("Material.Ka", 0.9f, 0.5f, 0.3f);
		m_solidShader.setVec3("Light.La", 0.4f, 0.4f, 0.4f);
		m_solidShader.setVec3("Material.Ks", 0.8f, 0.8f, 0.8f);
		m_solidShader.setVec3("Light.Ls", 1.0f, 1.0f, 1.0f);
		m_solidShader.setFloat("Material.Shininess", 100.0f);
	}

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, m_normVBO.getVBO());
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
	
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, m_colorVBO.getVBO());
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertVBO.getVBO());
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glDrawArrays(GL_TRIANGLES, 0, m_vertVBO.getSize());
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);

	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);

	if (m_bShowWireframe)
	{
		m_wireframeShader.disable();
	}
	else
	{
		m_solidShader.disable();
	}


    glPopAttrib();
}

void RigidTriangleRender::resize(unsigned int triNum)
{
	m_vertVBO.resize(3 * triNum);
	m_normVBO.resize(3 * triNum);
	m_colorVBO.resize(3 * triNum);
}


}//end of namespace Physika