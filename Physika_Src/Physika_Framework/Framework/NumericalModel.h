#pragma once
#include "Physika_Framework/Framework/Module.h"
#include "Physika_Framework/Framework/Base.h"

namespace Physika
{
	class NumericalModel : public Module
	{
	public:
		NumericalModel();
		~NumericalModel() override;

		virtual void step(Real dt) {};

		virtual void updateTopology() {};

		std::string getModuleType() override { return "NumericalModel"; }
	protected:
		
	private:

	};
}

