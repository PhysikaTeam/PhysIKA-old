#pragma once

#ifndef _SANDSOLVERINTERFACE_H
#define _SANDSOLVERINTERFACE_H

#include "Dynamics/Sand/types.h"

#include "Dynamics/Sand/SandGrid.h"

namespace PhysIKA {
class SandSolverInterface
{

public:
    virtual bool initialize()
    {
        return true;
    }

    virtual bool stepSimulation(float deltime) = 0;

    virtual float getMaxTimeStep() = 0;

    virtual void setSandGridInfo(const SandGridInfo& sandinfo)
    {
        m_SandInfo = sandinfo;
    }

    SandGridInfo& getSandGridInfo()
    {
        return m_SandInfo;
    }
    const SandGridInfo& getSandGridInfo() const
    {
        return m_SandInfo;
    }

    virtual void updateUserParticle(DeviceArray<Vector3f>& usePoints) {}

public:
    SandGridInfo m_SandInfo;
};
}  // namespace PhysIKA

#endif