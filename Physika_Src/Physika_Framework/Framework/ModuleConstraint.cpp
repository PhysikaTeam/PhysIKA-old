#include "ModuleConstraint.h"
#include "Physika_Framework/Framework/Node.h"

namespace Physika
{
ConstraintModule::ConstraintModule()
	: Module()
	, m_posID(MechanicalState::position())
	, m_velID(MechanicalState::velocity())
{
}

ConstraintModule::~ConstraintModule()
{
}

}