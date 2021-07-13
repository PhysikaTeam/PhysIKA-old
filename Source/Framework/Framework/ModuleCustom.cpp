#include "ModuleCustom.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {
IMPLEMENT_CLASS(CustomModule)

CustomModule::CustomModule()
    : Module()
{
}

CustomModule::~CustomModule()
{
}

bool CustomModule::execute()
{
    this->applyCustomBehavior();
    return true;
}

void CustomModule::applyCustomBehavior()
{
}

}  // namespace PhysIKA