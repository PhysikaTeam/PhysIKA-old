#pragma once

#include "SystemMotionState.h"
#include "Framework/Framework/Module.h"
#include "Dynamics/RigidBody/ForwardDynamicsSolver.h"
#include <memory>
namespace PhysIKA {
class FeatherstoneIntegrationModule : public Module
{
public:
    FeatherstoneIntegrationModule();

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

private:
    std::shared_ptr<ForwardDynamicsSolver> m_fd_solver;

    Vectornd<float> m_ddq;
    double          m_dt = 0;

    double m_last_time = 0;
    bool   m_time_init = false;
};
}  // namespace PhysIKA
