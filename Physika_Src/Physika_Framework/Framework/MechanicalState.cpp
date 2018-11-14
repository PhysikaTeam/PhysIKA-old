#include "MechanicalState.h"

namespace Physika
{
	MechanicalState::MechanicalState()
	{

	}

	MechanicalState::~MechanicalState(void)
	{

	}

	void MechanicalState::resetForce()
	{
		resetField(MechanicalState::force());
		resetField(MechanicalState::forceMoment());
		resetField(MechanicalState::d_force());
	}

	void MechanicalState::resetField(std::string name)
	{
		auto field = this->getField(name);
		if (field != nullptr)
		{
			field->reset();
		}
	}

}