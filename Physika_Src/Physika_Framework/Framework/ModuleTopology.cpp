#include "Framework/ModuleTopology.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(TopologyModule)

TopologyModule::TopologyModule()
	: Module()
	, m_topologyChanged(true)
{

}

TopologyModule::~TopologyModule()
{
}

}