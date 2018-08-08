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
	void insertToNodeImpl(Node* node) override;
	void deleteFromNodeImpl(Node* node) override;
private:

};
}

