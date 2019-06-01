#pragma once
#include "Framework/Framework/Module.h"
#include "Framework/Framework/CollidableObject.h"

namespace Physika
{

class CollisionModel : public Module
{
public:
	CollisionModel();
	virtual ~CollisionModel();

	virtual bool isSupport(std::shared_ptr<CollidableObject> obj) = 0;

	virtual void doCollision() = 0;
	
	std::string getModuleType() override { return "CollisionModel"; }

	virtual void addCollidableObject(std::shared_ptr<CollidableObject> obj) {};
protected:
};

}
