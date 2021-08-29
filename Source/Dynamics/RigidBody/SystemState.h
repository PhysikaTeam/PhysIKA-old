#pragma once
#include "SystemMotionState.h"
#include "Core/Vector/vector_nd.h"
#include "SpatialVector.h"
#include <memory>
#include <vector>

namespace PhysIKA {
class SystemState
{
public:
    SystemState(Node* node = 0)
        : m_root(node)
    {
    }

    void setRoot(Node* node = 0)
    {
        this->m_root = node;
        if (this->m_motionState)
            this->m_motionState->setRoot(node);
    }

    void setRigidNum(int n)
    {
        this->m_externalForce.resize(n);
        if (m_motionState)
        {
            m_motionState->setRigidNum(n);
        }
    }

    void setNum(int n, int dof)
    {
        this->m_externalForce.resize(n);
        this->m_activeForce.resize(dof);
        if (m_motionState)
        {
            m_motionState->setNum(n, dof);
        }
    }

public:
    Node* m_root = 0;

    std::shared_ptr<SystemMotionState> m_motionState = 0;

    std::vector<SpatialVector<float>> m_externalForce;  // The origin of the external force is the center of rigid body.

    Vectornd<float> m_activeForce;  //

    Vector3f m_gravity;
};
}  // namespace PhysIKA
