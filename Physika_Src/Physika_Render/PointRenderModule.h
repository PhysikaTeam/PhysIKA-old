#pragma once

#include "Physika_Framework/Framework/ModuleVisual.h"
#include "Physika_Render/PointRender.h"
#include "Physika_Render/LineRender.h"
#include "Physika_Render/TriangleRender.h"

namespace Physika
{
	class PointRenderModule : public VisualModule
	{
		DECLARE_CLASS(PointRenderModule)
	public:
		PointRenderModule();
		~PointRenderModule();

		enum RenderMode {
			POINT = 0,
			SPRITE
		};

		void display() override;
		void setRenderMode(RenderMode mode);
		void setColor(Vector3f color);

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		RenderMode m_mode;
		Vector3f m_color;

// 		std::shared_ptr<PointRenderUtil> point_render_util;
// 		std::shared_ptr<PointRenderTask> point_render_task;
		std::shared_ptr<PointRender> m_pointRender;
		std::shared_ptr<LineRender> m_lineRender;
		std::shared_ptr<TriangleRender> m_triangleRender;
	};

}