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
	virtual bool connectPosition(std::shared_ptr<Field>& pos) { return true; }
	virtual bool connectVelocity(std::shared_ptr<Field>& vel) { return true; }

	virtual void constrain() {};

	std::string getModuleType() override { return "ConstraintModule"; }
private:
};
}
