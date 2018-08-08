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

void TopologyModule::insertToNodeImpl(Node* node)
{
	node->setTopologyModule(this);
}

void TopologyModule::deleteFromNodeImpl(Node* node)
{

}

}