#pragma once

#include <GL/glew.h>
#include "PointRenderModule.h"
#include "Framework/Topology/PointSet.h"
#include "Core/Vector.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "OpenGLContext.h"
#include "Color.h"


namespace PhysIKA
{
	IMPLEMENT_CLASS(PointRenderModule)

	PointRenderModule::PointRenderModule()
		: VisualModule()
		, m_mode(PointRenderModule::Instance)
		, m_color(Vector3f(0.8, 0.8, 0.8))
	{
		m_minIndex.setValue(0);
		m_maxIndex.setValue(1);

		m_refV = m_minIndex.getValue();

		this->attachField(&m_minIndex, "minIndex", "minIndex", false);
		this->attachField(&m_maxIndex, "maxIndex", "maxIndex", false);

		this->attachField(&m_vecIndex, "vectorIndex", "vectorIndex", false);
		this->attachField(&m_scalarIndex, "scalarIndex", "scalarIndex", false);
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

		DeviceArray<float3>* xyz = (DeviceArray<float3>*)&(pSet->getPoints());

		m_pointRender = std::make_shared<PointRender>();
		m_pointRender->resize(xyz->size());
		m_colorArray.resize(xyz->size());

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

		Log::sendMessage(Log::Info, "PointRenderModule successfully initialized!");
	}

	__global__ void PRM_MappingColor(
		DeviceArray<glm::vec3> color,
		DeviceArray<Vector3f> index,
		float minIndex,
		float maxIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= color.size()) return;

		float index_i = index[tId].norm();

		index_i = index_i > maxIndex ? maxIndex : index_i;
		index_i = index_i < minIndex ? minIndex : index_i;

		float a = (index_i - minIndex) / (maxIndex - minIndex);

		Color hsv;
		hsv.HSVtoRGB(240, 1-a, 1);

		color[tId] = glm::vec3(hsv.r, hsv.g, hsv.b);
	}

	__global__ void PRM_MappingColor(
		DeviceArray<glm::vec3> color,
		DeviceArray<float> index,
		float refV,
		float minIndex,
		float maxIndex)
	{
		int tId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (tId >= color.size()) return;

		float index_i = index[tId];

		index_i = index_i > maxIndex ? maxIndex : index_i;
		index_i = index_i < minIndex ? minIndex : index_i;

		float a = (index_i - refV) / (maxIndex - minIndex);

		Color hsv;
		hsv.HSVtoRGB(a * 120 + 120, 1, 1);

		color[tId] = glm::vec3(hsv.r, hsv.g, hsv.b);
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
		
		if (!m_vecIndex.isEmpty())
		{
			uint pDims = cudaGridSize(xyz->size(), BLOCK_SIZE);
			PRM_MappingColor << <pDims, BLOCK_SIZE >> > (
				m_colorArray,
				m_vecIndex.getValue(),
				m_minIndex.getValue(),
				m_maxIndex.getValue());
			cuSynchronize();

			m_pointRender->setColor(m_colorArray);
		}
		else if (!m_scalarIndex.isEmpty())
		{
			uint pDims = cudaGridSize(xyz->size(), BLOCK_SIZE);
			PRM_MappingColor << <pDims, BLOCK_SIZE >> > (
				m_colorArray,
				m_scalarIndex.getValue(),
				m_refV,
				m_minIndex.getValue(),
				m_maxIndex.getValue());
			cuSynchronize();

			m_pointRender->setColor(m_colorArray);
		}
		else
		{
			m_pointRender->setColor(glm::vec3(m_color[0], m_color[1], m_color[2]));
		}

		
		//if (m_colorArray)
		//	m_pointRender->setColorArray(*(DeviceArray<float3>*)m_colorArray.get());
		
		m_pointRender->setVertexArray(*xyz);
	}

	void PointRenderModule::display()
	{
		glMatrixMode(GL_MODELVIEW_MATRIX);
		glPushMatrix();

		glRotatef(m_rotation.x(), m_rotation.y(), m_rotation.z(), m_rotation.w());
		glTranslatef(m_translation[0], m_translation[1], m_translation[2]);
		glScalef(m_scale[0], m_scale[1], m_scale[2]);

 		//m_pointRender->display();

		switch (m_mode)
		{
		case PointRenderModule::POINT:
			m_pointRender->renderPoints();
			break;
		case PointRenderModule::SPRITE:
			m_pointRender->renderSprite();
			break;
		case PointRenderModule::Instance:
			m_pointRender->renderInstancedSphere();
		default:
			break;
		}

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

	void PointRenderModule::setColorRange(float min, float max)
	{
		m_minIndex.setValue(min);
		m_maxIndex.setValue(max);
	}

	void PointRenderModule::setReferenceColor(float v)
	{
		m_refV = v;
	}

}