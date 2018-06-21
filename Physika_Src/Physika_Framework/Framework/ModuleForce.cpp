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

bool ForceModule::insertToNode(Node* node)
{
	node->addForceModule(this);
	return true;
}

bool ForceModule::deleteFromNode(Node* node)
{
	node->deleteForceModule(this);
	return true;
}

}