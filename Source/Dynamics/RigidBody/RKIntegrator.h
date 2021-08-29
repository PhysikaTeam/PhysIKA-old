#pragma once

#include "SystemMotionState.h"
namespace PhysIKA {
class RK4Integrator
{
public:
    //template<typename State, template DState>
    //State solve(const State& s0, void(*dydt)(const State&, DState&), double dt)
    //SystemMotionState solve(const State& s0, void(*dydt)(const State&, DState&), double dt)
    //void solve(SystemMotionState& s0, void(*dydt)(const SystemMotionState&, DSystemMotionState&), double dt)
    template <typename DYDT>
    void solve(SystemMotionState& s0, DYDT& dydt, double dt)
    {
        //return s0 + dydt(s0) * dt;

        SystemMotionState s;
        //DSystemMotionState k;

        s = s0;
        DSystemMotionState k1;
        dydt(s, k1);

        s = s0;
        s.addDs(k1, dt * 0.5);
        DSystemMotionState k2;
        dydt(s, k2);

        s = s0;
        s.addDs(k2, dt * 0.5);
        DSystemMotionState k3;
        dydt(s, k3);

        s = s0;
        s.addDs(k3, dt);
        DSystemMotionState k4;
        dydt(s, k4);

        double newdt = dt / 6.0;
        s0.addDs(k1, newdt);
        s0.addDs(k2, newdt * 2.0);
        s0.addDs(k3, newdt * 2.0);
        s0.addDs(k4, newdt);

        //DSystemMotionState k;
        //dydt(s0, k);
        //s0.addDs(k, dt);

        //return s0;// +(k1 + (k2*2.0) + (k3*2.0) + k4)*(dt / 6.0);
    }
};
}  // namespace PhysIKA
