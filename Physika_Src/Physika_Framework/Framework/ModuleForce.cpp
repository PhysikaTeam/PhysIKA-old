#include "ModuleForce.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(ForceModule)

ForceModule::ForceModule()
{
}

ForceModule::~ForceModule()
{
}

void ForceModule::insertToNodeImpl(Node* node)
{
	node->addForceModule(this);
}

void ForceModule::deleteFromNodeImpl(Node* node)
{
	node->deleteForceModule(this);
}

}