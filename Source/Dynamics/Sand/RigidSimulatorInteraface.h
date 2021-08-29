#pragma once
#include "vector_types.h"

//#include "RigidBody.h"
//#include "Dynamics/Sand/BulletDynamics/Dynamics/btRigidBody.h"
//#include "Framework/Framework/Node.h"

namespace PhysIKA {
class RigidSimulatorInterface
{
public:
    virtual void setRigidPosition(int rigididx, float px, float py, float pz)          = 0;
    virtual void setRigidRotation(int rigididx, float w, float qx, float qy, float qz) = 0;

    virtual float3 getRigidLinearV(int rigididx)                                = 0;
    virtual float3 getRigidAngularV(int rigididx)                               = 0;
    virtual void   setRigidLinearV(int rigididx, float vx, float vy, float vz)  = 0;
    virtual void   setRigidAngularV(int rigididx, float ax, float ay, float az) = 0;

    virtual void applyCentralImpulse(int rigididx, float px, float py, float pz)     = 0;
    virtual void applyTorqueImpulse(int rigididx, float tx, float ty, float tz)      = 0;
    virtual void applyRigidImpulse(int rigididx, const float3& ip, const float3& it) = 0;

    virtual void setRigidLinearVelocity(int rigididx, const float3& lv)  = 0;
    virtual void setRigidAngularVeclotiy(int rigididx, const float3& av) = 0;

    /**
        * @brief Set velocity threshold.
        * @details Before this function, all objects should be created.
        */
    virtual void setVelocityThreshold(float linvThreshold, float angvThreshold) = 0;

    virtual void setGravity(float gx, float gy, float gz)
    {
        m_gravity.x = gx;
        m_gravity.y = gy;
        m_gravity.z = gz;
    }
    virtual void applyGravity(float timeStep) = 0;

    virtual void stepSimulation(float timeStep) = 0;

    //virtual btRigidBody* getRigidBody(int rigidId) = 0;
    //virtual int addRigidBody(btRigidBody* prigid) = 0;

public:
    // std::vector<RigidBody*> m_sseRigids;

    float3 m_gravity = { 0, 0, 0 };
};

}  // namespace PhysIKA