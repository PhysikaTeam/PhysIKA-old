#pragma once

#include "Framework/Framework/Module.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Dynamics/RigidBody/ForwardDynamicsSolver.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"
//#include "Dynamics/RigidBody/RigidState.h"

#include "Dynamics/RigidBody/ArticulatedBodyFDSolver.h"
#include "ForwardDynamicsSolver.h"

#include <memory>
#include <vector>

namespace PhysIKA {

class RigidTimeIntegrationModule : public Module
{
    DECLARE_CLASS(RigidTimeIntegrationModule)

public:
public:
    RigidTimeIntegrationModule();

    bool initialize(){};

    virtual void begin();

    virtual bool execute();

    virtual void end(){};

    //virtual void updateSystemState(double dt);

    //virtual void updateSystemState(const RigidState& s);

    void setDt(double dt)
    {
        m_dt = dt;
    }

    void dydt(const SystemMotionState& s0, DSystemMotionState& ds);

    void dydt(const SystemState& s0, const SystemMotionState& motionState, DSystemMotionState& ds);

    void setFDSolver(std::shared_ptr<ForwardDynamicsSolver> fd_solver);

    std::shared_ptr<ForwardDynamicsSolver> getFDSolver()
    {
        return m_fd_solver;
    }

private:
    std::shared_ptr<ForwardDynamicsSolver> m_fd_solver;

    Vectornd<float> m_ddq;
    double          m_dt = 0;

    double m_last_time = 0;
    bool   m_time_init = false;
};

class DydtAdapter
{
public:
    DydtAdapter(RigidTimeIntegrationModule* integrator = 0)
        : m_integrator(integrator)
    {
    }

    void setIntegrator(RigidTimeIntegrationModule* integrator)
    {
        m_integrator = integrator;
    }

    void operator()(const SystemMotionState& s0, DSystemMotionState& ds)
    {
        if (m_integrator)
        {
            m_integrator->dydt(s0, ds);
        }
    }

public:
    RigidTimeIntegrationModule* m_integrator;
};

}  // namespace PhysIKA