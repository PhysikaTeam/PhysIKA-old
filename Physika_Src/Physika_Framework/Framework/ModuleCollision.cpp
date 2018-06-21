#include "ModuleCollision.h"
#include "Framework/Node.h"

namespace Physika
{
IMPLEMENT_CLASS(CollisionModule)

CollisionModule::CollisionModule()
{
}

CollisionModule::~CollisionModule()
{
}

bool CollisionModule::insertToNode(Node* node)
{
	node->addCollisionModule(this);
	return true;
}

bool CollisionModule::deleteFromNode(Node* node)
{
	node->deleteCollisionModule(this);
	return true;
}

}