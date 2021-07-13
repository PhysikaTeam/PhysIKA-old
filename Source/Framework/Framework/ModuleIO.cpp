#include "Framework/Framework/ModuleIO.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {
IOModule::IOModule()
    : Module()
    , m_enabled(true)
{
}

IOModule::~IOModule()
{
}

}  // namespace PhysIKA