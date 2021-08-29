#pragma once
#include "Framework/Framework/Module.h"

namespace PhysIKA {
class CollidableObject : public Module
{
public:
    enum CType
    {
        SPHERE_TYPE = 0,
        TRIANGLE_TYPE,
        TETRAHEDRON_TYPE,
        POINTSET_TYPE,
        SIGNED_DISTANCE_TYPE,
        UNDFINED
    };

public:
    CollidableObject(CType ctype);
    virtual ~CollidableObject();

    CType getType()
    {
        return m_type;
    }

    //should be called before collision is started
    virtual void updateCollidableObject() = 0;

    //should be called after the collision is finished
    virtual void updateMechanicalState() = 0;

    std::string getModuleType() override
    {
        return "CollidableObject";
    }

private:
    CType m_type;
};
}  // namespace PhysIKA
