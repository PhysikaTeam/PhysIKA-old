#include "Joint.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "PrismaticJoint.h"
namespace PhysIKA {
//IMPLEMENT_CLASS_1(Joint, TDataType)

PrismaticJoint::PrismaticJoint(std::string name)
    : Joint(name)
{
}

PrismaticJoint::PrismaticJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
}

void PrismaticJoint::setJointInfo(const Vector3f& mv_dir)
{
    //m_S[0][0] = 0; m_S[1][0] = 0; m_S[2][0] = 0;

    m_S(3, 0) = mv_dir[0];
    m_S(4, 0) = mv_dir[1];
    m_S(5, 0) = mv_dir[2];

    this->m_S.getBases()[0].normalize();
}
}  // namespace PhysIKA