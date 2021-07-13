#pragma once
#include "Framework/Framework/Node.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Dynamics/RigidBody/Joint.h"

#include <memory>

namespace PhysIKA {
/**
    * @brief Joint of vehicle front wheels
    * @details The joint has 2 rotation freedom.
    */

class VehicleFrontJoint : public Joint
{
    //DECLARE_CLASS_1(Joint, TDataType)
public:
    VehicleFrontJoint(std::string name = "default");

    VehicleFrontJoint(Node* predecessor, Node* successor);

    // Get degree of freedom of joint
    virtual int getJointDOF() const
    {
        return 2;
    }

    // Get matrix of JointSpace.
    virtual const JointSpaceBase<float>& getJointSpace() const
    {
        return this->m_S;
    }

    // Set matrix of JointSpace.
    virtual void setJointSpace(const JointSpaceBase<float>& S)
    {
        this->m_S = dynamic_cast<const JointSpace<float, 2>&>(S);
    }  // { m_S = S; }

    // set relative motion direction, expressed in successor frame
    // mv_dir: motion direction, dim = 3
    /**
        * @brief Set relative motion direction, expressed in successor frame
        * @param rollingAxis, steeringAxis
        */
    void setJointInfo(const Vector3f& rollingAxis, const Vector3f& steeringAxis);

private:
    JointSpace<float, 2> m_S;
};
}  // namespace PhysIKA