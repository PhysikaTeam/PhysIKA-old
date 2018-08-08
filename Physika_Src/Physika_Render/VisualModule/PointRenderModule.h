#pragma once

#include "Physika_Framework/Framework/ModuleVisual.h"
#include "Physika_Render/Point_Render/point_render_util.h"
#include "Physika_Render/Point_Render/point_gl_cuda_buffer.h"
#include "Physika_Render/Point_Render/point_render_task.h"

namespace Physika
{
	class PointRenderModule : public VisualModule
	{
	public:
		PointRenderModule();
		~PointRenderModule();

		bool  initialize() override;

		void display() override;

	private:
		std::shared_ptr<PointRenderUtil> point_render_util;
		std::shared_ptr<PointRenderTask> point_render_task;
	};

}