#include "Module.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(Module)

Module::Module()
{

}

Module::Module(Node* node)
{
	m_node = node;
}

Module::~Module(void)
{
	m_ins.clear();
	m_outs.clear();
}

std::shared_ptr<DeviceContext> Module::getContext()
{
	return m_node->getContext();
}

}