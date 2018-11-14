#pragma once
#include "Framework/Module.h"
#include "Framework/Base.h"

namespace Physika
{
	class NumericalModel : public Module
	{
	public:
		NumericalModel();
		~NumericalModel() override;

		virtual void step(Real dt) {};

		virtual void updateTopology() = 0;

		std::string getModuleType() override { return "NumericalModel"; }
	protected:
		
	private:

	};
}

