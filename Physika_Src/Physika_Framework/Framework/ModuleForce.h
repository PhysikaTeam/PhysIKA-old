#pragma once
#include "Framework/Module.h"

namespace Physika{

class ForceModule : public Module
{
	DECLARE_CLASS(ForceModule)
public:
	ForceModule();
	virtual ~ForceModule();

public:
	bool insertToNode(Node* node) override;
	bool deleteFromNode(Node* node) override;
private:

};
}

