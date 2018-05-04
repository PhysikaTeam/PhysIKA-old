#pragma once
#include "Framework/Module.h"

namespace Physika
{

class RenderModule : public Module
{
	DECLARE_CLASS(RenderModule)
public:
	RenderModule();
	virtual ~RenderModule();

	virtual void Display() {};

private:
	bool insertToNode(Node* node) override;
	bool deleteFromNode(Node* node) override;
};

}
