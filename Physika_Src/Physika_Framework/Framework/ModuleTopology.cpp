#include "Physika_Framework/Framework/ModuleTopology.h"
#include "Physika_Framework/Framework/Node.h"

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