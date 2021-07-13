#include "FreeJoint.h"
#include "Dynamics/RigidBody/RigidUtil.h"

namespace PhysIKA {
//IMPLEMENT_CLASS_1(FreeJoint, TDataType)

PhysIKA::FreeJoint::FreeJoint(std::string name)
    : Joint(name)
{

    this->m_S(0, 0) = 1;
    this->m_S(1, 1) = 1;
    this->m_S(2, 2) = 1;
    this->m_S(3, 3) = 1;
    this->m_S(4, 4) = 1;
    this->m_S(5, 5) = 1;
}

FreeJoint::FreeJoint(Node* predecessor, Node* successor)
    : Joint(predecessor, successor)
{
    this->m_S(0, 0) = 1;
    this->m_S(1, 1) = 1;
    this->m_S(2, 2) = 1;
    this->m_S(3, 3) = 1;
    this->m_S(4, 4) = 1;
    this->m_S(5, 5) = 1;
}

//FreeJoint::~FreeJoint()
//{

//}

//
//void FreeJoint::setRotateAxis(const Vectornd<float>& axis)
//{
//    Vectornd<float> unit_axis = axis.normalized();
//
//    // calculate motion subspace matrix S.
//    this->m_S.resize(6, 1);
//    this->m_S(0, 0) = unit_axis(0);
//    this->m_S(1, 0) = unit_axis(1);
//    this->m_S(2, 0) = unit_axis(2);

//    // calculate constraint force subspace T
//    RigidUtil::calculateOrthogonalSpace(this->m_S, this->m_T);

//}

}  // namespace PhysIKA