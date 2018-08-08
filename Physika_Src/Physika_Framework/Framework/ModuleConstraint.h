#pragma once
#include "Framework/Module.h"

namespace Physika
{
class Field;

class ConstraintModule : public Module
{
public:
	ConstraintModule();
	virtual ~ConstraintModule();

	//interface for data initialization, must be called before execution
	virtual bool connectPosition(std::shared_ptr<Field>& pos) = 0;
	virtual bool connectVelocity(std::shared_ptr<Field>& vel) = 0;


	void insertToNodeImpl(Node* node) override;
	void deleteFromNodeImpl(Node* node) override;
private:
};
}
