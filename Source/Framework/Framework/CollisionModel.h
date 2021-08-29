#pragma once
#include "Framework/Framework/Module.h"
#include "Framework/Framework/CollidableObject.h"

namespace PhysIKA {

class ContactPair
{
public:
    int id[2];

    Real m_stiffness;
    Real m_friction;
};

class CollisionModel : public Module
{
public:
    CollisionModel();
    virtual ~CollisionModel();

    virtual bool isSupport(std::shared_ptr<CollidableObject> obj) = 0;

    bool execute() override;

    virtual void doCollision() = 0;

    std::string getModuleType() override
    {
        return "CollisionModel";
    }

    virtual void addCollidableObject(std::shared_ptr<CollidableObject> obj){};

protected:
};

}  // namespace PhysIKA
