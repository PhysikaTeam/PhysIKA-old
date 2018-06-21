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
	bool insertToNode(Node* node) override;
	bool deleteFromNode(Node* node) override;
};

}
