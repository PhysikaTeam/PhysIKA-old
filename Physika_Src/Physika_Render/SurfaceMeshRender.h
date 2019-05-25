#pragma once

#include "Physika_Framework/Framework/ModuleVisual.h"
#include "Physika_Render/PointRender.h"
#include "Physika_Render/LineRender.h"
#include "Physika_Render/TriangleRender.h"

namespace Physika
{
	class SurfaceMeshRender : public VisualModule
	{
		DECLARE_CLASS(SurfaceMeshRender)
	public:
		SurfaceMeshRender();
		~SurfaceMeshRender();

		void display() override;
		void setColor(Vector3f color);

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		Vector3f m_color;

		DeviceArray<float3> vertices;
		DeviceArray<float3> normals;
		DeviceArray<float3> colors;

		std::shared_ptr<TriangleRender> m_triangleRender;
	};

}