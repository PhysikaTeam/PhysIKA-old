#include "ModuleConstraint.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(ConstraintModule)

ConstraintModule::ConstraintModule()
{
}

ConstraintModule::~ConstraintModule()
{
}

bool ConstraintModule::insertToNode(Node* node)
{
	node->addConstraintModule(this);
	return true;
}

bool ConstraintModule::deleteFromNode(Node* node)
{
	node->deleteConstraintModule(this);
	return true;
}

}