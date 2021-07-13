#pragma once
#include "Framework/Framework/Module.h"

namespace PhysIKA {
class Field;

class ConstraintModule : public Module
{
public:
    ConstraintModule();
    ~ConstraintModule() override;

    //interface for data initialization, must be called before execution
    virtual bool connectPosition(Field*& pos)
    {
        return true;
    }
    virtual bool connectVelocity(Field*& vel)
    {
        return true;
    }

    void setPositionID(FieldID id)
    {
        m_posID = id;
    }
    void setVelocityID(FieldID id)
    {
        m_velID = id;
    }

    virtual bool constrain()
    {
        return true;
    }

    std::string getModuleType() override
    {
        return "ConstraintModule";
    }

protected:
    FieldID m_posID;
    FieldID m_velID;
};
}  // namespace PhysIKA
