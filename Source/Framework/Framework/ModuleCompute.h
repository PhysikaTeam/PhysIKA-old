#pragma once
#include "Framework/Framework/Module.h"

namespace PhysIKA {

class ComputeModule : public Module
{
public:
    ComputeModule();
    ~ComputeModule() override;

    bool execute() override;

    virtual void compute(){};

    std::string getModuleType() override
    {
        return "ComputeModule";
    }

private:
};
}  // namespace PhysIKA
