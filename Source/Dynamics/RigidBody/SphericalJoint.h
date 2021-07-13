#pragma once

#include "Dynamics/RigidBody/Joint.h"

//#include "Dynamics/RigidBody/RigidBody2.h"
#include <memory>

namespace PhysIKA {

/*!
    *    \class    Spherical joint
    *    \brief    Joint of fixed point rotation.
    *
    *    This kind of joint has no constraint on rigid bodies.
    *    But it is useful for manage the system.
    *
    */

class SphericalJoint : public Joint
{

public:
    SphericalJoint(std::string name = "default");

    SphericalJoint(Node* predecessor, Node* successor);
    //virtual ~FreeJoint();

    // Get degree of freedom of joint
    virtual int getJointDOF() const
    {
        return 3;
    }
    // Get matrix of JointSpace.
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return this->m_S;
    }
    // Set matrix of JointSpace.
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        this->m_S = dynamic_cast<const JointSpace<float, 3>&>(S);
    }  // { m_S = S; }

    /**
        * @brief Set joint information. 
        * @details Setup joint space matrix according to position of joint
        * @param r Joint position. Reltive to successor, in successor frame.
        * @return void
        */
    virtual void setJointInfo(const Vector3f& r);

private:
    JointSpace<float, 3> m_S;
};

//#ifdef PRECISION_FLOAT
//    template class FreeJoint<DataType3f>;
//#else
//    template class FreeJoint<DataType3d>;
//#endif
}  // namespace PhysIKA