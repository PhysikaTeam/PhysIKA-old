#include "Joint.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "CylindricalJoint.h"

namespace PhysIKA {
//IMPLEMENT_CLASS_1(Joint, TDataType)

CylindricalJoint::CylindricalJoint(std::string name)
    : Joint(name)
{
}

CylindricalJoint::CylindricalJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
}

void CylindricalJoint::setJointInfo(const VectorBase<float>& axis)
{
    //Vectornd<float> unit_axis(axis.normalized());

    //this->m_constrain_dim = 2;
    //this->m_S.resize(6, 2);

    m_S(0, 0) = axis[0];
    m_S(1, 0) = axis[1];
    m_S(2, 0) = axis[2];

    m_S(3, 1) = axis[0];
    m_S(4, 1) = axis[1];
    m_S(5, 1) = axis[2];

    //RigidUtil::calculateOrthogonalSpace(this->m_S, this->m_T);
}
}  // namespace PhysIKA