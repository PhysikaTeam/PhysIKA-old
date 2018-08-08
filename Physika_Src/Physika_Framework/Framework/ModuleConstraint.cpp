#include "ModuleConstraint.h"
#include "Framework/Node.h"

namespace Physika
{
ConstraintModule::ConstraintModule()
{
}

ConstraintModule::~ConstraintModule()
{
}

void ConstraintModule::insertToNodeImpl(Node* node)
{
	node->addConstraintModule(this);
}

void ConstraintModule::deleteFromNodeImpl(Node* node)
{
	node->deleteConstraintModule(this);
}

}