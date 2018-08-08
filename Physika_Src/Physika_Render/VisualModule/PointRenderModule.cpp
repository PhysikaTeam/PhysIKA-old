#pragma once
#include "PointRenderModule.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Framework/Framework/Node.h"

namespace Physika
{
	PointRenderModule::PointRenderModule()
		: VisualModule()
	{
	}

	PointRenderModule::~PointRenderModule()
	{
	}

	bool PointRenderModule::initialize()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		PointSet<Vector3f>* pSet = dynamic_cast<PointSet<Vector3f>*>(parent->getTopologyModule());
		if (pSet == NULL)
		{
			Log::sendMessage(Log::Error, "The topology module is not supported!");
			return false;
		}

		point_render_util = std::make_shared<PointRenderUtil>();

		DeviceArray<float3>* xyz = (DeviceArray<float3>*)pSet->getPoints();
		PointGLCudaBuffer point_gl_cuda_buffer = point_render_util->mapPointGLCudaBuffer(xyz->size());

		point_render_util->unmapPointGLCudaBuffer();

		//---------------------------------------------------------------------------------------------------------------------
		point_render_task = std::make_shared<PointRenderTask>(point_render_util);
		//point_render_task->disableUsePointSprite();
		point_render_task->setPointScaleForPointSprite(3.0);
	}

	void PointRenderModule::display()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return;
		}

		PointSet<Vector3f>* pSet = dynamic_cast<PointSet<Vector3f>*>(parent->getTopologyModule());
		if (pSet == NULL)
		{
			Log::sendMessage(Log::Error, "The topology module is not supported!");
			return;
		}

		DeviceArray<float3>* xyz = (DeviceArray<float3>*)pSet->getPoints();
		PointGLCudaBuffer point_gl_cuda_buffer = point_render_util->mapPointGLCudaBuffer(xyz->size());
		cudaMemcpy(point_gl_cuda_buffer.getCudaPosPtr(), xyz->getDataPtr(), sizeof(float) * 3 * xyz->size(), cudaMemcpyDeviceToDevice);
		point_render_util->unmapPointGLCudaBuffer();

		point_render_task->renderTask();
	}

}