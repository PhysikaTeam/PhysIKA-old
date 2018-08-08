#include "Framework/ModuleVisual.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(VisualModule)

VisualModule::VisualModule()
	: Module()
{
}

VisualModule::~VisualModule()
{
}

void VisualModule::insertToNodeImpl(Node* node)
{
	node->addVisualModule(this);
}

void VisualModule::deleteFromNodeImpl(Node* node)
{
	node->deleteVisualModule(this);
}

}