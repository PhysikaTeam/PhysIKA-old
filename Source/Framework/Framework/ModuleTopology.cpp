#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {
IMPLEMENT_CLASS(TopologyModule)

TopologyModule::TopologyModule()
    : Module()
    , m_topologyChanged(true)
{
}

TopologyModule::~TopologyModule()
{
}

}  // namespace PhysIKA