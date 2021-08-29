#pragma once
#include "Framework/Framework/Module.h"

namespace PhysIKA {

class CustomModule : public Module
{
    DECLARE_CLASS(CustomModule)
public:
    CustomModule();
    virtual ~CustomModule();

    bool execute() override;

    std::string getModuleType() override
    {
        return "CustomModule";
    }

protected:
    virtual void applyCustomBehavior();
};
}  // namespace PhysIKA
