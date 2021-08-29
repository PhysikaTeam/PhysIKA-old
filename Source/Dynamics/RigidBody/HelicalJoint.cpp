#include "Joint.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "HelicalJoint.h"
namespace PhysIKA {
//IMPLEMENT_CLASS_1(Joint, TDataType)

HelicalJoint::HelicalJoint(std::string name)
    : Joint(name)
{
}

HelicalJoint::HelicalJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
}

void HelicalJoint::setJointInfo(const VectorBase<float>& axis, float h)
{
    //Vectornd<float> unit_axis = axis.normalized();
    //this->m_constrain_dim = 1;
    //this->m_S.resize(6, 1);

    // should axis be normalized?
    // should m_S be normalized?

    m_S(0, 0) = axis[0];
    m_S(1, 0) = axis[1];
    m_S(2, 0) = axis[2];
    m_S(3, 0) = axis[0] * h;
    m_S(4, 0) = axis[1] * h;
    m_S(5, 0) = axis[2] * h;

    this->m_S.getBases()[0].normalize();

    //RigidUtil::calculateOrthogonalSpace(this->m_S, this->m_T);
}
}  // namespace PhysIKA