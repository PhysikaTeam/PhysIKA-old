#pragma once

#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/Base.h"
#include <queue>
#include <memory>
#include <vector>
#include "SystemState.h"

#include "Core/Utility/CTimer.h"
#include <iostream>

namespace PhysIKA {
/**
    * @brief Interface of forward dynamcis solver.
    * @details Forward dynamics solver calculates system acceleration according to system force and motion state.
    */
class ForwardDynamicsSolver : public Base
{
public:
    Node* getParent()
    {
        return m_parent_node;
    }
    void setParent(Node* parent_node)
    {
        m_parent_node = parent_node;
    }

    /**
        * @brief Solve forward dynamcis
        * @param s_system Current system state. The force state will be used.
        * @param s Current system motion state. Position and velocity information will be used.
        * @param ddq The vector to save the results.
        * @return Success or not
        *    @retval Success or not
        */
    virtual bool solve(const SystemState& s_system, const SystemMotionState& s, Vectornd<float>& ddq)
    {
        return true;
    };

protected:
    Node* m_parent_node = 0;
};

class InertiaMatrixFDSolver : public ForwardDynamicsSolver
{
public:
    DECLARE_CLASS(InertiaMatrixFDSolver)

    InertiaMatrixFDSolver();

    void buildJointSpaceMotionEquation(const SystemState& s_system, const SystemMotionState& s, MatrixMN<float>& H, Vectornd<float>& C);

    bool solve(const SystemState& s_system, const SystemMotionState& s, Vectornd<float>& ddq);

    //static void dydt(const SystemState& s_system, const SystemMotionState& s, SystemMotionState& ds);

private:
    Node* m_parent_node = 0;

    MatrixMN<float> m_H;
    Vectornd<float> m_C;
};

}  // namespace PhysIKA