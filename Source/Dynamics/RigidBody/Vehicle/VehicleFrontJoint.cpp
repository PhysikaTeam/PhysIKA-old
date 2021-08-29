#include "Dynamics/RigidBody/Joint.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "VehicleFrontJoint.h"
namespace PhysIKA {
//IMPLEMENT_CLASS_1(Joint, TDataType)

VehicleFrontJoint::VehicleFrontJoint(std::string name)
    : Joint(name)
{
}

VehicleFrontJoint::VehicleFrontJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
}

void VehicleFrontJoint::setJointInfo(const Vector3f& rollingAxis, const Vector3f& steeringAxis)
{
    //Vector3f axis1 = rollingAxis;
    m_S(3, 0) = 0;
    m_S(4, 0) = 0;
    m_S(5, 0) = 0;
    m_S(0, 0) = rollingAxis[0];
    m_S(1, 0) = rollingAxis[1];
    m_S(2, 0) = rollingAxis[2];

    m_S(3, 1) = 0;
    m_S(4, 1) = 0;
    m_S(5, 1) = 0;
    m_S(0, 1) = steeringAxis[0];
    m_S(1, 1) = steeringAxis[1];
    m_S(2, 1) = steeringAxis[2];

    m_S.getBases()[0].normalize();
    m_S.getBases()[1].normalize();
}

//void VehicleFrontJoint::setJointInfo(const Vector3f& mv_dir)
//{
//
//    m_S(3,0) = mv_dir[0];
//    m_S(4,0) = mv_dir[1];
//    m_S(5,0) = mv_dir[2];
//
//    this->m_S.getBases()[0].normalize();
//}
}  // namespace PhysIKA