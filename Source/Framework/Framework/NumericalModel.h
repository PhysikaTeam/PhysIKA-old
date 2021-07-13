#pragma once
#include "Framework/Framework/Module.h"
#include "Framework/Framework/Base.h"

namespace PhysIKA {
class NumericalModel : public Module
{
public:
    NumericalModel();
    ~NumericalModel() override;

    virtual void step(Real dt){};

    virtual void updateTopology(){};

    std::string getModuleType() override
    {
        return "NumericalModel";
    }

protected:
private:
};
}  // namespace PhysIKA
