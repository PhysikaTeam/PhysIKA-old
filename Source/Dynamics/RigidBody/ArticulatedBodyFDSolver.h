#pragma once

#include "ForwardDynamicsSolver.h"

#include <vector>
#include "Core/Matrix/matrix_mxn.h"
#include "SpatialVector.h"
#include "JointSpace.h"

namespace PhysIKA {
/**
    * @brief Interface of forward dynamcis solver.
    * @details Forward dynamics solver calculates system acceleration according to system force and motion state.
    */
class ArticulatedBodyFDSolver : public ForwardDynamicsSolver
{
public:
    ArticulatedBodyFDSolver()
        : m_isValid(false)
    {
    }

    /**
        * @brief Solve forward dynamcis
        * @param s_system Current system state. The force state will be used.
        * @param s Current system motion state. Position and velocity information will be used.
        * @param ddq The vector to save the results.
        * @return Success or not.
        *    @retval Success or not.
        */
    virtual bool solve(const SystemState& s_system, const SystemMotionState& s, Vectornd<float>& ddq);

    void setValidtiy(bool isValid)
    {
        m_isValid = isValid;
    }

    virtual void init();

private:
    std::vector<SpatialVector<float>> m_pA;
    std::vector<MatrixMN<float>>      m_IA;
    std::vector<JointSpace<float, 6>> m_U;
    std::vector<MatrixMN<float>>      m_D_inv;
    Vectornd<float>                   m_ui;
    std::vector<SpatialVector<float>> m_a;

    bool m_isValid = false;
};

}  // namespace PhysIKA