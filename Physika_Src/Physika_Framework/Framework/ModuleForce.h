#pragma once
#include "Physika_Framework/Framework/Module.h"

namespace Physika{

class ForceModule : public Module
{
	DECLARE_CLASS(ForceModule)
public:
	ForceModule();
	virtual ~ForceModule();

	virtual void applyForce() {};

	std::string getModuleType() override { return "ForceModule"; }
private:

};
}

