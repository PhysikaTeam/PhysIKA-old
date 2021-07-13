#include "ModuleConstraint.h"
#include "Framework/Framework/Node.h"

namespace PhysIKA {
ConstraintModule::ConstraintModule()
    : Module()
    , m_posID(MechanicalState::position())
    , m_velID(MechanicalState::velocity())
{
}

ConstraintModule::~ConstraintModule()
{
}

}  // namespace PhysIKA