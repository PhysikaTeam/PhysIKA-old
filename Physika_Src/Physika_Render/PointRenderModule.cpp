#pragma once

#include <GL/glew.h>
#include "PointRenderModule.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Framework/Framework/Node.h"
#include "OpenGLContext.h"


namespace Physika
{
	IMPLEMENT_CLASS(PointRenderModule)

	PointRenderModule::PointRenderModule()
		: VisualModule()
		, m_mode(PointRenderModule::SPRITE)
		, m_color(Vector3f(0.8, 0.8, 0.8))
	{
	}

	PointRenderModule::~PointRenderModule()
	{
	}

	bool PointRenderModule::initializeImpl()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(parent->getTopologyModule());
		if (pSet == nullptr)
		{
			Log::sendMessage(Log::Error, "PointRenderModule: The topology module is not supported!");
			return false;
		}

		if (!pSet->isInitialized())
		{
			pSet->initialize();
		}

//		point_render_util = std::make_shared<PointRenderUtil>();

		DeviceArray<float3>* xyz = (DeviceArray<float3>*)&(pSet->getPoints());
//		PointGLCudaBuffer point_gl_cuda_buffer = point_render_util->mapPointGLCudaBuffer(xyz->size());

//		point_render_util->unmapPointGLCudaBuffer();

		//---------------------------------------------------------------------------------------------------------------------
		//point_render_task = std::make_shared<PointRenderTask>(point_render_util);
		//point_render_task->disableUsePointSprite();
		//point_render_task->setPointScaleForPointSprite(3.0);

		m_pointRender = std::make_shared<PointRender>();
		m_pointRender->resize(xyz->size());

		switch (m_mode)
		{
		case PointRenderModule::POINT:
			m_pointRender->disableUsePointSprite();
			break;
		case PointRenderModule::SPRITE:
			m_pointRender->enableUsePointSprite();
			break;
		default:
			break;
		}

// 		m_lineRender = std::make_shared<LineRender>();
// 		m_lineRender->resize(xyz->size()/2);
// 		m_lineRender->setLines(*xyz);
// 
// 		HostArray<float3> color;
// 		HostArray<float3> normals;
// 		color.resize(xyz->size());
// 		normals.resize(xyz->size());
// 
// 		for (int i = 0; i < xyz->size(); i++)
// 		{
// 			color[i].x = rand() % 1000 / 1000.0;
// 			color[i].y = rand() % 1000 / 1000.0;
// 			color[i].z = rand() % 1000 / 1000.0;
// 			normals[i].x = 1.0f;
// 			normals[i].y = 0.0f;
// 			normals[i].z = 0.0f;
// 		}
// 
// 		m_lineRender->setColors(color);
// 
// 		m_triangleRender = std::make_shared<TriangleRender>();
// 		m_triangleRender->resize(floor(xyz->size() / 3.0));
// 		m_triangleRender->setVertexArray(*xyz);
// 		m_triangleRender->setColorArray(color);
// 		m_triangleRender->setNormalArray(normals);
	}

	void PointRenderModule::updateRenderingContext()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return;
		}

		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(parent->getTopologyModule());
		if (pSet == nullptr)
		{
			Log::sendMessage(Log::Error, "PointRenderModule: The topology module is not supported!");
			return;
		}

		DeviceArray<float3>* xyz = (DeviceArray<float3>*)&(pSet->getPoints());


		m_pointRender->setColor(glm::vec3(m_color[0], m_color[1], m_color[2]));
		m_pointRender->setVertexArray(*xyz);

		
	}

	void PointRenderModule::display()
	{
		glMatrixMode(GL_MODELVIEW_MATRIX);
		glPushMatrix();

		glRotatef(m_rotation.x(), m_rotation.y(), m_rotation.z(), m_rotation.w());
		glTranslatef(m_translation[0], m_translation[1], m_translation[2]);
		glScalef(m_scale[0], m_scale[1], m_scale[2]);

 		m_pointRender->display();

		glPopMatrix();
	}

	void PointRenderModule::setRenderMode(RenderMode mode)
	{
		m_mode = mode;
	}

	void PointRenderModule::setColor(Vector3f color)
	{
		m_color = color;
	}

}