#pragma once
#include "Framework/Framework/Module.h"

namespace Physika
{
class TopologyMapping : public Module
{
public:
	TopologyMapping();
	virtual ~TopologyMapping();

	virtual bool apply() = 0;
private:

};

}