#include "ModuleCompute.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {
ComputeModule::ComputeModule()
{
}

ComputeModule::~ComputeModule()
{
}

bool ComputeModule::execute()
{
    this->compute();

    return true;
}

}  // namespace PhysIKA