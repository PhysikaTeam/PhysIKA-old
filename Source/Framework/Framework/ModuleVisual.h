#pragma once
#include "Framework/Framework/Module.h"
#include "Core/Vector.h"
#include "Core/Quaternion/quaternion.h"
#include "FieldVar.h"

namespace PhysIKA {

class VisualModule : public Module
{
public:
    VisualModule();
    virtual ~VisualModule();

    virtual void display(){};

    void setVisible(bool bVisible);
    bool isVisible()
    {
        return m_visible.getValue();
    }

    void rotate(float angle, float x, float y, float z);
    void translate(float x, float y, float z);
    void scale(float x, float y, float z);

    virtual void updateRenderingContext(){};

    std::string getModuleType() override
    {
        return "VisualModule";
    }

protected:
    VarField<bool> m_visible;

    Quaternion<float> m_rotation;
    Vector3f          m_scale;
    Vector3f          m_translation;
};

}  // namespace PhysIKA
