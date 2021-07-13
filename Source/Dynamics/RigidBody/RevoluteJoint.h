#pragma once
#include "Framework/Framework/Node.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"

#include "Dynamics/RigidBody/Joint.h"

//#include "Dynamics/RigidBody/RigidBody2.h"
#include <memory>

namespace PhysIKA {
//template<typename TDataType> class Frame;
/*!
    *    \class    RigidBody
    *    \brief    Rigid body dynamics.
    *
    *    This class implements a simple rigid body.
    *
    */

class RevoluteJoint : public Joint
{
    //DECLARE_CLASS_1(RevoluteJoint, TDataType)
public:
    RevoluteJoint(std::string name = "default");

    RevoluteJoint(Node* predecessor, Node* successor);

    // Get degree of freedom of joint
    virtual int getJointDOF() const
    {
        return 1;
    }
    // Get matrix of JointSpace.
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return this->m_S;
    }
    // Set matrix of JointSpace.
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        this->m_S = dynamic_cast<const JointSpace<float, 1>&>(S);
    }  // { m_S = S; }

    /**
        * @brief Set joint rotation axis and joint position
        * @details Joint position and rotation axis should be expressed in successor frame. Joint position is relative to successor frame origin.
        * @param axis Joint rotation axis, expressed in successor frame.
        * @param r Joint position, relative to successor frame origin, expressed in successor frame.
        * @return void
        */
    void setJointInfo(const Vector3f& axis, const Vector3f& r);

private:
    JointSpace<float, 1> m_S;
};

}  // namespace PhysIKA