#include "Framework/Framework/ModuleTopology.h"
#include "Framework/Framework/Node.h"

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