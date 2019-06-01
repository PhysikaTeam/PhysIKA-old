#pragma once
#include "Framework/Framework/Module.h"

namespace Physika{

class ComputeModule : public Module
{
public:
	ComputeModule();
	~ComputeModule() override;

	virtual void compute() {};

	std::string getModuleType() override { return "ComputeModule"; }
private:

};
}

