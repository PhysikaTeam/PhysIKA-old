#pragma once
#include "Framework/Module.h"
#include "Framework/Base.h"

namespace Physika
{
	class State
	{
	public:
		static std::string position() { return "Position"; }
		static std::string velocity() { return "Velocity"; }
		static std::string acceleration() { return "Acceleration"; }
		static std::string force() { return "Force"; }
	};

	class NumericalModel : public Module
	{
		DECLARE_CLASS(NumericalModel)
	public:
		NumericalModel();
		~NumericalModel() override;

	protected:

	private:

	};
}

