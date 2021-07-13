#pragma once

#include "Framework/Framework/ModuleVisual.h"
#include "Rendering/PointRender.h"
#include "Rendering/LineRender.h"
#include "Rendering/RigidTriangleRender.h"
#include "Framework/Topology/Frame.h"
#include "Core/DataTypes.h"

namespace PhysIKA {
class RigidMeshRender : public VisualModule
{
    DECLARE_CLASS(RigidMeshRender)
public:
    RigidMeshRender(std::shared_ptr<Frame<DataType3f>> frame = 0);
    ~RigidMeshRender();

    inline void setMeshFrame(std::shared_ptr<Frame<DataType3f>> frame)
    {
        this->m_meshFrame = frame;
    }

    void                                 display() override;
    void                                 setColor(Vector3f color);
    std::shared_ptr<RigidTriangleRender> getTriangleRender()
    {
        return m_triangleRender;
    }

    void setTriangleRotation(const Quaternion<float>& q)
    {
        m_triangleRender->setRotation(q);
    }
    void setTriangleTranslation(const Vector3f& t)
    {
        m_triangleRender->setTranslatioin(t);
    }
    void setTriangleScale(const Vector3f& s)
    {
        m_triangleRender->setScale(s);
    }

protected:
    bool initializeImpl() override;

    void updateRenderingContext() override;

private:
    Vector3f m_color;

    DeviceArray<float3> vertices;
    DeviceArray<float3> normals;
    DeviceArray<float3> colors;

    std::shared_ptr<RigidTriangleRender> m_triangleRender;
    std::shared_ptr<Frame<DataType3f>>   m_meshFrame;
};

}  // namespace PhysIKA