#pragma once

#include "Framework/Framework/Module.h"

#include "Core/Vector/vector_nd.h"

#include "Framework/Framework/Node.h"

#include "Dynamics/RigidBody/Transform3d.h"
#include "SpatialVector.h"

#include <memory>

namespace PhysIKA {
struct DSystemMotionState
{
public:
    void setRigidNum(int n)
    {
        m_rel_r.resize(n);
        m_rel_q.resize(n);
        m_v.resize(n);
    }

    void setDof(int n)
    {
        m_dq.resize(n);
        m_generalq.resize(n);
    }

    Vectornd<float>& getGeneralFreedom()
    {
        return m_generalq;
    }
    const Vectornd<float>& getGeneralFreedom() const
    {
        return m_generalq;
    }

    std::vector<SpatialVector<float>>& getGlobalVelocity()
    {
        return globalVelocity;
    }
    const std::vector<SpatialVector<float>>& getGlobalVelocity() const
    {
        return globalVelocity;
    }

public:
    std::vector<Vector3f>          m_rel_r;  // relative position
    std::vector<Quaternion<float>> m_rel_q;  // relative rotation

    std::vector<SpatialVector<float>> m_v;  // Relative spatial velocities

    std::vector<SpatialVector<float>> globalVelocity;  // Global velocity in world frame.

    Vectornd<float> m_dq;
    Vectornd<float> m_generalq;
};

struct SystemMotionState  //:public State
{
public:
    SystemMotionState(Node* root = 0)
        : m_root(root)
    {
        //build();
    }

    void setRoot(Node* root)
    {
        m_root = root;
    }

    //void build();
    SystemMotionState& addDs(const DSystemMotionState& ds, double dt);

    void updateGlobalInfo();

    void setRigidNum(int n)
    {

        m_rel_r.resize(n);
        m_rel_q.resize(n);
        globalPosition.resize(n);
        globalRotation.resize(n);
        m_X.resize(n);
        m_v.resize(n);

        globalVelocity.resize(n);
    }

    void setNum(int n, int dof)
    {
        m_rel_r.resize(n);
        m_rel_q.resize(n);
        globalPosition.resize(n);
        globalRotation.resize(n);
        m_X.resize(n);
        m_v.resize(n);

        generalVelocity.resize(dof);
        generalPosition.resize(dof);

        globalVelocity.resize(n);
    }

    //static void dydt(const SystemMotionState& s0, SystemMotionState& ds);

    std::vector<Vector3f>& getGlobalPosition()
    {
        return globalPosition;
    }
    const std::vector<Vector3f>& getGlobalPosition() const
    {
        return globalPosition;
    }

    std::vector<Quaternion<float>>& getGlobalRotation()
    {
        return globalRotation;
    }
    const std::vector<Quaternion<float>>& getGlobalRotation() const
    {
        return globalRotation;
    }

    Vectornd<float>& getGeneralPosition()
    {
        return generalPosition;
    }
    const Vectornd<float>& getGeneralPosition() const
    {
        return generalPosition;
    }

    Vectornd<float>& getGeneralVelocity()
    {
        return generalVelocity;
    }
    const Vectornd<float>& getGeneralVelocity() const
    {
        return generalVelocity;
    }

    std::vector<SpatialVector<float>>& getGlobalVelocity()
    {
        return globalVelocity;
    }
    const std::vector<SpatialVector<float>>& getGlobalVelocity() const
    {
        return globalVelocity;
    }

private:
public:
    Node* m_root = 0;

    // ---------x
    std::vector<Vector3f>          m_rel_r;  // relative position in parent frame.
    std::vector<Quaternion<float>> m_rel_q;

    std::vector<Vector3f>          globalPosition;  // global positions of rigid bodies
    std::vector<Quaternion<float>> globalRotation;  // global rotations of rigid bodies

    std::vector<Transform3d<float>> m_X;  // Transformations from parent nodes to child nodes

    // -------- v
    std::vector<SpatialVector<float>> m_v;  // Relative spatial velocities, successor frame

    std::vector<SpatialVector<float>> globalVelocity;  // Global velocity in world frame.

    // General velocities in joint space.
    // For eanch joint, its dof can be 0-6.
    // It will be inefficient to use a Vectornd, as we need to allocate dynamic memory for eanch Vectornd.
    Vectornd<float> generalVelocity;  // General velocities in joint space.
    Vectornd<float> generalPosition;  // General displacement.
};

//class RigidSystemFor
}  // namespace PhysIKA