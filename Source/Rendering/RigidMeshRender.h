#pragma once

#include "Framework/Framework/ModuleVisual.h"
#include "Rendering/PointRender.h"
#include "Rendering/LineRender.h"
#include "Rendering/RigidTriangleRender.h"

namespace PhysIKA
{
	class RigidMeshRender : public VisualModule
	{
		DECLARE_CLASS(RigidMeshRender)
	public:
		RigidMeshRender();
		~RigidMeshRender();

		void display() override;
		void setColor(Vector3f color);
		std::shared_ptr<RigidTriangleRender> getTriangleRender() { return m_triangleRender; }

		void setTriangleRotation(const Quaternion<float>& q) { m_triangleRender->setRotation(q); }
		void setTriangleTranslation(const Vector3f& t) { m_triangleRender->setTranslatioin(t); }
		void setTriangleScale(const Vector3f& s) { m_triangleRender->setScale(s); }

	protected:
		bool  initializeImpl() override;

		void updateRenderingContext() override;

	private:
		Vector3f m_color;

		DeviceArray<float3> vertices;
		DeviceArray<float3> normals;
		DeviceArray<float3> colors;

		std::shared_ptr<RigidTriangleRender> m_triangleRender;
	};

}