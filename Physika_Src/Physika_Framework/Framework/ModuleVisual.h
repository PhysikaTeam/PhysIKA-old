#pragma once
#include "Framework/Module.h"

namespace Physika
{

class VisualModule : public Module
{
	DECLARE_CLASS(VisualModule)
public:
	VisualModule();
	virtual ~VisualModule();

	virtual void display() {};

private:
	bool insertToNode(Node* node) override;
	bool deleteFromNode(Node* node) override;
};

}
