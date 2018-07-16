#include "Framework/ModuleVisual.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(VisualModule)

VisualModule::VisualModule()
{
}

VisualModule::~VisualModule()
{
}

bool VisualModule::insertToNode(Node* node)
{
	node->addVisualModule(this);
	return true;
}

bool VisualModule::deleteFromNode(Node* node)
{
	node->deleteVisualModule(this);
	return true;
}

}