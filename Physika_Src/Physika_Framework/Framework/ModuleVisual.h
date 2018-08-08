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
	void insertToNodeImpl(Node* node) override;
	void deleteFromNodeImpl(Node* node) override;
};

}
