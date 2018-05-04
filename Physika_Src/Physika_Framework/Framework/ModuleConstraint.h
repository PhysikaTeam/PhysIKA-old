#pragma once
#include "Framework/Module.h"

namespace Physika
{
class ConstraintModule : public Module
{
	DECLARE_CLASS(ConstraintModule)
public:
	ConstraintModule();
	virtual ~ConstraintModule();

	bool insertToNode(Node* node) override;
	bool deleteFromNode(Node* node) override;
private:

};
}
