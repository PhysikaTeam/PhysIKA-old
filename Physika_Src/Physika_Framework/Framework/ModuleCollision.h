#pragma once
#include "Framework/Module.h"

namespace Physika
{

class CollisionModule : public Module
{
	DECLARE_CLASS(CollisionModule)
public:
	CollisionModule();
	virtual ~CollisionModule();

private:
	void insertToNodeImpl(Node* node) override;
	void deleteFromNodeImpl(Node* node) override;
};

}
