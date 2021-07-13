#pragma once
#include "Framework/Framework/Node.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Dynamics/RigidBody/Joint.h"
//#include "Dynamics/RigidBody/RigidBody2.h"
#include <memory>

namespace PhysIKA {
/**
    *    class: PrismaticJoint
    *    Jont that can only move in certain direction.
    *    Dof: 1
    */

class PrismaticJoint : public Joint
{
    //DECLARE_CLASS_1(Joint, TDataType)
public:
    PrismaticJoint(std::string name = "default");

    PrismaticJoint(Node* predecessor, Node* successor);

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

    // set relative motion direction, expressed in successor frame
    // mv_dir: motion direction, dim = 3
    void setJointInfo(const Vector3f& mv_dir);

private:
    JointSpace<float, 1> m_S;
};
}  // namespace PhysIKA