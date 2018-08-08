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

void CollisionModule::insertToNodeImpl(Node* node)
{
	node->addCollisionModule(this);
}

void CollisionModule::deleteFromNodeImpl(Node* node)
{
	node->deleteCollisionModule(this);
}

}