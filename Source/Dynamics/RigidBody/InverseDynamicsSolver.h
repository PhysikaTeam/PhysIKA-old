#pragma once

#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Framework/Framework/Node.h"
#include "SystemState.h"

#include <memory>
//#include<vector>

namespace PhysIKA {
/**
    * class:  InverseDynamicSolver
    * brief:  Inverse dynamic solver will be used to solve inverse dynamic problem.
    *         An inverse dynamic problemis defined as:  t = ID(model, q, dq, ddq)
    *         That is solving active force according to given acceleration.
    */
class InverseDynamicsSolver
{
public:
    InverseDynamicsSolver(Node* parent_node = 0);

    /// solve inverse dynamic problem
    /// Input: s, system state;
    ///        ddq, general acceleration
    ///        zeroAcceleration, denote wether the general acceleration is zero.
    ///        tau, the result active force.
    //void inverseDynamics(const SystemState& s, const Vectornd<float>& ddq, Vectornd<float>& tau, bool zeroAcceleration = false);
    void inverseDynamics(const SystemState& s, const Vectornd<float>& ddq, Vectornd<float>& tau, bool zeroAcceleration = false);

    Node* getParent()
    {
        return m_parent_node;
    }
    void setParent(Node* parent_node)
    {
        m_parent_node = parent_node;
    }

private:
    Node* m_parent_node = 0;
};

}  // namespace PhysIKA